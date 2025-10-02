# ekf_cbf_demo
# EKF Localization + CBF-QP Safe Navigation (Python)

This project demonstrates **Extended Kalman Filter (EKF) localization** combined with a **Control Barrier Function (CBF) Quadratic Program (QP) safety filter** for a mobile robot navigating to a goal while avoiding obstacles.

The robot:
- Uses noisy range/bearing measurements from landmarks
- Estimates its pose with an EKF
- Navigates safely to the goal using a CBF-QP safety controller
- Visualizes the trajectory in a Matplotlib animation

## 🚀 Run the demo

```bash
pip install -r requirements.txt
python main.py

