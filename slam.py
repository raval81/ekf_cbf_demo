# EKF-localization + CBF-QP safety demo
# - Simple unicycle robot
# - Known point landmarks (range-bearing measurements)
# - EKF state: [x, y, theta]
# - Use EKF estimate for controller + CBF
# - Uses cvxopt for QP

import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

# ---------- QP solver ----------
def cbf_qp(u_nom, A, b, u_min, u_max):
    n = len(u_nom)
    P = 2 * matrix(np.eye(n))
    q = -2 * matrix(u_nom)
    G_list, h_list = [], []

    if A is not None:
        G_list.append(-A.reshape(1, -1))
        h_list.append(b)

    # box constraints
    G_list.append(np.eye(n))
    h_list.append(u_max)
    G_list.append(-np.eye(n))
    h_list.append(-u_min)

    G = matrix(np.vstack(G_list))
    h = matrix(np.hstack(h_list))
    sol = solvers.qp(P, q, G, h)
    return np.array(sol['x']).flatten()

# ---------- Simulation and EKF ----------
dt = 0.1
steps = 300

# True robot state
x_true = np.array([0.0, 0.0, 0.0])

# EKF estimate (initial)
x_est = np.array([0.0, 0.0, 0.0])
P = np.diag([0.01, 0.01, 0.01])  # small initial uncertainty

# Motion noise (control noise) covariance (for prediction)
Q = np.diag([0.02, 0.02, 0.01])

# Measurement noise: [range_var, bearing_var]
R_range = 0.05**2
R_bearing = (np.deg2rad(2.0))**2

# Landmarks (known positions for localization)
landmarks = np.array([
    [1.0, 4.0],
    [4.5, 1.0],
    [3.0, 4.0],
    [0.5, 2.5]
])

# Obstacle (circular)
obs = np.array([2.5, 2.5])
obs_radius = 1.0
safe_dist = 0.6

# Limits
u_min = np.array([0.0, -1.5])
u_max = np.array([1.2, 1.5])

# Goal
goal = np.array([5.0, 5.0])

# Histories
true_traj, est_traj, ctrl_nom_hist, ctrl_safe_hist = [], [], [], []

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

for k in range(steps):
    # ---------- SENSE (simulate noisy landmark measurements) ----------
    measurements = []  # list of (idx, range, bearing)
    for i, lm in enumerate(landmarks):
        dx = lm[0] - x_true[0]
        dy = lm[1] - x_true[1]
        r = np.hypot(dx, dy)
        if r < 6.0:  # sensor range
            bearing = np.arctan2(dy, dx) - x_true[2]
            # add noise
            r_meas = r + np.random.randn()*np.sqrt(R_range)
            b_meas = wrap_angle(bearing + np.random.randn()*np.sqrt(R_bearing))
            measurements.append((i, r_meas, b_meas))

    # ---------- NOMINAL CONTROLLER (uses estimate) ----------
    # diff = goal - x_est[:2]
    # dist_goal = np.linalg.norm(diff)
    # theta_d = np.arctan2(diff[1], diff[0])
    # v_nom = 1.1 * np.tanh(1.2*dist_goal)  # saturating approach
    # w_nom = 3.0 * wrap_angle(theta_d - x_est[2])

    # # Add tangential repulsion near obstacle (based on estimate)
    # dx_obs = x_est[0] - obs[0]
    # dy_obs = x_est[1] - obs[1]
    # dist_obs = np.hypot(dx_obs, dy_obs)
    # if dist_obs < (obs_radius + safe_dist + 0.5):
    #     # aim to add tangential heading
    #     rep_angle = np.arctan2(dy_obs, dx_obs) + np.pi/2
    #     # mix into angular command
    #     w_nom += 1.2 * wrap_angle(rep_angle - x_est[2])
    #     # reduce forward slightly
    #     v_nom *= 0.9

    # u_nom = np.array([v_nom, w_nom])
# # ---------- SMART NOMINAL CONTROLLER ----------
#     # Artificial potential field: attraction to goal + repulsion from obstacle
#     diff = goal - x_est[:2]
#     dist_goal = np.linalg.norm(diff)
    
#     # Attractive force toward goal
#     k_att = 1.5
#     f_att = k_att * diff / (dist_goal + 0.01)
    
#     # Repulsive force from obstacle
#     dx_obs = x_est[0] - obs[0]
#     dy_obs = x_est[1] - obs[1]
#     dist_obs = np.hypot(dx_obs, dy_obs)
#     d_influence = obs_radius + safe_dist + 1.5  # influence range
    
#     if dist_obs < d_influence:
#         k_rep = 1.0
#         f_rep = k_rep * (1.0/dist_obs - 1.0/d_influence) * (1.0/dist_obs**2) * np.array([dx_obs, dy_obs]) / dist_obs
#     else:
#         f_rep = np.array([0.0, 0.0])
    
#     # Total desired direction
#     f_total = f_att + f_rep
#     desired_angle = np.arctan2(f_total[1], f_total[0])
    
#     # Control commands
#     v_nom = 1.0 * np.tanh(0.8 * dist_goal)
#     w_nom = 3.0 * wrap_angle(desired_angle - x_est[2])
#     u_nom = np.array([v_nom, w_nom])
# ---------- IMPROVED NAVIGATION CONTROLLER ----------
    diff = goal - x_est[:2]
    dist_goal = np.linalg.norm(diff)
    
    # Obstacle info
    dx_obs = x_est[0] - obs[0]
    dy_obs = x_est[1] - obs[1]
    dist_obs = np.hypot(dx_obs, dy_obs)
    
    # Goal attraction (stronger when far from obstacle)
    goal_angle = np.arctan2(diff[1], diff[0])
    
    # If near obstacle, add tangential component to "slide around"
    d_influence = obs_radius + safe_dist + 1.2
    if dist_obs < d_influence:
        # Vector from obstacle to robot
        radial = np.array([dx_obs, dy_obs]) / dist_obs
        # Tangent vectors (perpendicular to radial)
        tangent1 = np.array([-radial[1], radial[0]])
        tangent2 = np.array([radial[1], -radial[0]])
        
        # Choose tangent that aligns better with goal direction
        goal_dir = diff / (dist_goal + 0.01)
        if np.dot(tangent1, goal_dir) > np.dot(tangent2, goal_dir):
            tangent = tangent1
        else:
            tangent = tangent2
        
        # Blend goal direction with tangent (more tangent when closer)
        weight_tangent = 0.7 * (1.0 - (dist_obs - (obs_radius + safe_dist)) / (d_influence - (obs_radius + safe_dist)))
        weight_tangent = np.clip(weight_tangent, 0, 0.8)
        
        desired_dir = (1 - weight_tangent) * goal_dir + weight_tangent * tangent
        desired_dir = desired_dir / (np.linalg.norm(desired_dir) + 1e-6)
        desired_angle = np.arctan2(desired_dir[1], desired_dir[0])
    else:
        desired_angle = goal_angle
    
    # Control commands
    v_nom = 0.9 * np.tanh(0.8 * dist_goal)
    w_nom = 2.5 * wrap_angle(desired_angle - x_est[2])
    u_nom = np.array([v_nom, w_nom])
    
    # ---------- CBF constraint ----------
    r_safe_total = obs_radius + safe_dist
    h = dist_obs**2 - r_safe_total**2
    
    cos_theta = np.cos(x_est[2])
    sin_theta = np.sin(x_est[2])
    A = np.array([[2 * (dx_obs * cos_theta + dy_obs * sin_theta), 0.0]])
    alpha = 0.6
    b = np.array([alpha * h])
    # ---------- CBF constraint ----------
    r_safe_total = obs_radius + safe_dist
    h = dist_obs**2 - r_safe_total**2
    
    cos_theta = np.cos(x_est[2])
    sin_theta = np.sin(x_est[2])
    A = np.array([[2 * (dx_obs * cos_theta + dy_obs * sin_theta), 0.0]])
    alpha = 0.8
    b = np.array([alpha * h])
    # ---------- CBF constraint (based on estimate) ----------
    # h = ||p - p_obs||^2 - (r_safe)^2
    r_safe_total = obs_radius + safe_dist
    dx_e = x_est[0] - obs[0]
    dy_e = x_est[1] - obs[1]
    dist_to_obs = np.hypot(dx_e, dy_e)
    #h = dx_e**2 + dy_e**2 - r_safe_total**2
    h = dist_to_obs**2 - r_safe_total**2
    grad_h = np.array([2*dx_e, 2*dy_e, 0.0])  # dh/dx
    radial_unit = np.array([dx_e / dist_to_obs, dy_e / dist_to_obs])
    cos_theta = np.cos(x_est[2])
    sin_theta = np.sin(x_est[2])

    # dynamics influence: v affects x,y; w affects theta
    f_v = np.array([np.cos(x_est[2]), np.sin(x_est[2]), 0.0])
    f_w = np.array([0.0, 0.0, 1.0])
    #A = np.array([grad_h @ f_v, grad_h @ f_w]).reshape(1, -1)
    A = np.array([[2 * (dx_e * cos_theta + dy_e * sin_theta), 0.0]])
    alpha = 0.5
    b = np.array([alpha * h])

    # Solve QP for safe control
    try:
        u_safe = cbf_qp(u_nom, A, b, u_min, u_max)
    except Exception as e:
        # if infeasible, fallback gracefully
        u_safe = np.clip(u_nom, u_min, u_max)

    # ---------- APPLY CONTROL TO TRUE ROBOT (simulate motion with noise) ----------
    v_true = u_safe[0] + np.random.randn()*0.01
    w_true = u_safe[1] + np.random.randn()*0.01
    x_true[0] += v_true * np.cos(x_true[2]) * dt
    x_true[1] += v_true * np.sin(x_true[2]) * dt
    x_true[2] = wrap_angle(x_true[2] + w_true * dt)

    # ---------- EKF PREDICT (on estimate) ----------
    # motion model: x' = x + v cosθ dt, y' = y + v sinθ dt, θ' = θ + w dt
    v_pred, w_pred = u_safe  # we feed applied control
    theta = x_est[2]
    # Jacobian F_x and F_u
    Fx = np.eye(3)
    Fx[0,2] = -v_pred * np.sin(theta) * dt
    Fx[1,2] = v_pred * np.cos(theta) * dt

    Fu = np.zeros((3,2))
    Fu[0,0] = np.cos(theta) * dt
    Fu[1,0] = np.sin(theta) * dt
    Fu[2,1] = dt

    # Predict state
    x_est = x_est + np.array([v_pred*np.cos(theta)*dt,
                              v_pred*np.sin(theta)*dt,
                              w_pred*dt])
    x_est[2] = wrap_angle(x_est[2])

    # Predict covariance
    P = Fx @ P @ Fx.T + Fu @ np.diag([0.01, 0.01]) @ Fu.T + Q

    # ---------- EKF UPDATE (for each measurement) ----------
    for (i, r_meas, b_meas) in measurements:
        lm = landmarks[i]
        dxm = lm[0] - x_est[0]
        dym = lm[1] - x_est[1]
        q = dxm**2 + dym**2
        z_hat_r = np.sqrt(q)
        z_hat_b = wrap_angle(np.arctan2(dym, dxm) - x_est[2])

        # measurement Jacobian H (range & bearing)
        H = np.zeros((2,3))
        H[0,0] = -dxm / z_hat_r
        H[0,1] = -dym / z_hat_r
        H[0,2] = 0.0
        H[1,0] =  dym / q
        H[1,1] = -dxm / q
        H[1,2] = -1.0

        R = np.diag([R_range, R_bearing])
        z = np.array([r_meas, b_meas])
        z_hat = np.array([z_hat_r, z_hat_b])
        y = z - z_hat
        y[1] = wrap_angle(y[1])

        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        delta = K @ y
        x_est = x_est + delta
        x_est[2] = wrap_angle(x_est[2])
        P = (np.eye(3) - K @ H) @ P

    # store histories
    true_traj.append(x_true.copy())
    est_traj.append(x_est.copy())
    ctrl_nom_hist.append(u_nom.copy())
    ctrl_safe_hist.append(u_safe.copy())

    # stopping if close to goal (true pos)
    if np.linalg.norm(x_true[:2] - goal) < 0.15:
        print("Reached goal at step", k)
        break

true_traj = np.array(true_traj)
est_traj = np.array(est_traj)

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(7,7))
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 6)
ax.set_aspect('equal')

# obstacle safety circle
r_safe_total = obs_radius + safe_dist
ax.add_patch(plt.Circle(obs, r_safe_total, color='r', alpha=0.3))

# landmarks
ax.scatter(landmarks[:,0], landmarks[:,1], c='orange', marker='s', label='Landmarks')

# goal
ax.plot(goal[0], goal[1], 'g*', markersize=15, label='Goal')

# true and estimated trajectories
ax.plot(true_traj[:,0], true_traj[:,1], 'b-', label='True traj')
ax.plot(est_traj[:,0], est_traj[:,1], 'b--', alpha=0.7, label='Estimated traj')

ax.plot(true_traj[0,0], true_traj[0,1], 'ko', label='Start (true)')
ax.plot(true_traj[-1,0], true_traj[-1,1], 'bo', label='End (true)')

ax.legend()
ax.set_title("EKF Localization + CBF-QP Safe Navigation")
plt.show()
