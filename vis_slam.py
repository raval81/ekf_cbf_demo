# EKF-localization + CBF-QP safety demo with ANIMATION
# Google Colab compatible version

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FuncAnimation
# from IPython.display import HTML
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

x_true = np.array([0.0, 0.0, 0.0])
x_est = np.array([0.0, 0.0, 0.0])
P = np.diag([0.01, 0.01, 0.01])
Q = np.diag([0.02, 0.02, 0.01])
R_range = 0.05**2
R_bearing = (np.deg2rad(2.0))**2

landmarks = np.array([
    [1.0, 4.0],
    [4.5, 1.0],
    [3.0, 4.0],
    [0.5, 2.5]
])

obs = np.array([2.5, 2.5])
obs_radius = 1.0
safe_dist = 0.6

u_min = np.array([0.0, -1.5])
u_max = np.array([1.2, 1.5])

goal = np.array([5.0, 5.0])

true_traj, est_traj = [], []

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

print("Running simulation...")
for k in range(steps):
    measurements = []
    for i, lm in enumerate(landmarks):
        dx = lm[0] - x_true[0]
        dy = lm[1] - x_true[1]
        r = np.hypot(dx, dy)
        if r < 6.0:
            bearing = np.arctan2(dy, dx) - x_true[2]
            r_meas = r + np.random.randn()*np.sqrt(R_range)
            b_meas = wrap_angle(bearing + np.random.randn()*np.sqrt(R_bearing))
            measurements.append((i, r_meas, b_meas))

    diff = goal - x_est[:2]
    dist_goal = np.linalg.norm(diff)
    
    dx_obs = x_est[0] - obs[0]
    dy_obs = x_est[1] - obs[1]
    dist_obs = np.hypot(dx_obs, dy_obs)
    
    goal_angle = np.arctan2(diff[1], diff[0])
    
    d_influence = obs_radius + safe_dist + 1.2
    if dist_obs < d_influence:
        radial = np.array([dx_obs, dy_obs]) / dist_obs
        tangent1 = np.array([-radial[1], radial[0]])
        tangent2 = np.array([radial[1], -radial[0]])
        
        goal_dir = diff / (dist_goal + 0.01)
        tangent = tangent1 if np.dot(tangent1, goal_dir) > np.dot(tangent2, goal_dir) else tangent2
        
        weight_tangent = 0.7 * (1.0 - (dist_obs - (obs_radius + safe_dist)) / (d_influence - (obs_radius + safe_dist)))
        weight_tangent = np.clip(weight_tangent, 0, 0.8)
        
        desired_dir = (1 - weight_tangent) * goal_dir + weight_tangent * tangent
        desired_dir = desired_dir / (np.linalg.norm(desired_dir) + 1e-6)
        desired_angle = np.arctan2(desired_dir[1], desired_dir[0])
    else:
        desired_angle = goal_angle
    
    v_nom = 0.9 * np.tanh(0.8 * dist_goal)
    w_nom = 2.5 * wrap_angle(desired_angle - x_est[2])
    u_nom = np.array([v_nom, w_nom])
    
    r_safe_total = obs_radius + safe_dist
    h = dist_obs**2 - r_safe_total**2
    
    cos_theta = np.cos(x_est[2])
    sin_theta = np.sin(x_est[2])
    A = np.array([[2 * (dx_obs * cos_theta + dy_obs * sin_theta), 0.0]])
    alpha = 0.5
    b = np.array([alpha * h])

    try:
        u_safe = cbf_qp(u_nom, A, b, u_min, u_max)
    except:
        u_safe = np.clip(u_nom, u_min, u_max)

    v_true = u_safe[0] + np.random.randn()*0.01
    w_true = u_safe[1] + np.random.randn()*0.01
    x_true[0] += v_true * np.cos(x_true[2]) * dt
    x_true[1] += v_true * np.sin(x_true[2]) * dt
    x_true[2] = wrap_angle(x_true[2] + w_true * dt)

    v_pred, w_pred = u_safe
    theta = x_est[2]
    Fx = np.eye(3)
    Fx[0,2] = -v_pred * np.sin(theta) * dt
    Fx[1,2] = v_pred * np.cos(theta) * dt

    Fu = np.zeros((3,2))
    Fu[0,0] = np.cos(theta) * dt
    Fu[1,0] = np.sin(theta) * dt
    Fu[2,1] = dt

    x_est = x_est + np.array([v_pred*np.cos(theta)*dt, v_pred*np.sin(theta)*dt, w_pred*dt])
    x_est[2] = wrap_angle(x_est[2])
    P = Fx @ P @ Fx.T + Fu @ np.diag([0.01, 0.01]) @ Fu.T + Q

    for (i, r_meas, b_meas) in measurements:
        lm = landmarks[i]
        dxm = lm[0] - x_est[0]
        dym = lm[1] - x_est[1]
        q = dxm**2 + dym**2
        z_hat_r = np.sqrt(q)
        z_hat_b = wrap_angle(np.arctan2(dym, dxm) - x_est[2])

        H = np.zeros((2,3))
        H[0,0] = -dxm / z_hat_r
        H[0,1] = -dym / z_hat_r
        H[1,0] = dym / q
        H[1,1] = -dxm / q
        H[1,2] = -1.0

        R = np.diag([R_range, R_bearing])
        z = np.array([r_meas, b_meas])
        z_hat = np.array([z_hat_r, z_hat_b])
        y = z - z_hat
        y[1] = wrap_angle(y[1])

        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x_est = x_est + K @ y
        x_est[2] = wrap_angle(x_est[2])
        P = (np.eye(3) - K @ H) @ P

    true_traj.append(x_true.copy())
    est_traj.append(x_est.copy())

    if np.linalg.norm(x_true[:2] - goal) < 0.15:
        print(f"Goal reached at step {k}!")
        break

true_traj = np.array(true_traj)
est_traj = np.array(est_traj)

print(f"Creating animation with {len(true_traj)} frames...")

# ---------- Create Animation ----------
fig, ax = plt.subplots(figsize=(9, 9))
plt.close()  # Don't show static figure

ax.set_xlim(-1, 6)
ax.set_ylim(-1, 6)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--')

# Static elements
r_safe_total = obs_radius + safe_dist
obstacle = Circle(obs, r_safe_total, color='red', alpha=0.3, label='Safety zone')
ax.add_patch(obstacle)

ax.scatter(landmarks[:,0], landmarks[:,1], c='orange', marker='s', s=150, 
           label='Landmarks', zorder=5, edgecolors='black', linewidth=2)
ax.plot(goal[0], goal[1], 'g*', markersize=25, label='Goal', zorder=5, 
        markeredgecolor='darkgreen', markeredgewidth=2)
ax.plot(true_traj[0,0], true_traj[0,1], 'ko', markersize=12, label='Start', zorder=5)

# Dynamic elements
true_path, = ax.plot([], [], 'b-', linewidth=3, label='True trajectory')
est_path, = ax.plot([], [], 'c--', linewidth=2, alpha=0.7, label='Estimated trajectory')

# Fix: Use facecolor instead of color for Polygon
robot_body = Polygon([[0,0]], facecolor='blue', alpha=0.9, edgecolor='darkblue', linewidth=2)
ax.add_patch(robot_body)

est_body = Polygon([[0,0]], facecolor='cyan', alpha=0.5, edgecolor='darkcyan', linewidth=1.5)
ax.add_patch(est_body)

info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                    verticalalignment='top', fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.95),
                    family='monospace')

ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
ax.set_title("EKF Localization + CBF-QP Safe Navigation", fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel("X Position (m)", fontsize=12, fontweight='bold')
ax.set_ylabel("Y Position (m)", fontsize=12, fontweight='bold')

def create_triangle(x, y, theta, size=0.2):
    x1 = x + size * np.cos(theta)
    y1 = y + size * np.sin(theta)
    x2 = x + size * np.cos(theta + 2.6)
    y2 = y + size * np.sin(theta + 2.6)
    x3 = x + size * np.cos(theta - 2.6)
    y3 = y + size * np.sin(theta - 2.6)
    return np.array([[x1, y1], [x2, y2], [x3, y3]])

def init():
    true_path.set_data([], [])
    est_path.set_data([], [])
    robot_body.set_xy([[0,0]])
    est_body.set_xy([[0,0]])
    info_text.set_text('')
    return true_path, est_path, robot_body, est_body, info_text

def animate(frame):
    true_path.set_data(true_traj[:frame+1, 0], true_traj[:frame+1, 1])
    est_path.set_data(est_traj[:frame+1, 0], est_traj[:frame+1, 1])
    
    if frame < len(true_traj):
        x_t, y_t, theta_t = true_traj[frame]
        x_e, y_e, theta_e = est_traj[frame]
        
        robot_body.set_xy(create_triangle(x_t, y_t, theta_t, size=0.20))
        est_body.set_xy(create_triangle(x_e, y_e, theta_e, size=0.16))
        
        dist = np.linalg.norm(true_traj[frame, :2] - goal)
        pos_error = np.linalg.norm(true_traj[frame, :2] - est_traj[frame, :2])
        info_text.set_text(f'Frame: {frame:3d}/{len(true_traj)-1}\nGoal dist: {dist:.2f} m\nEKF error: {pos_error:.3f} m')
    
    return true_path, est_path, robot_body, est_body, info_text

anim = FuncAnimation(fig, animate, init_func=init, 
                     frames=len(true_traj), interval=50, 
                     blit=True, repeat=True)

print("Rendering animation for Colab...")
# This is the key line for Google Colab - converts animation to HTML5 video
# HTML(anim.to_jshtml())
# ---------- Save or show animation ----------
anim.save("slam_animation.mp4", fps=20, dpi=150)   # save as video
plt.show()  # or directly show interactive animation
