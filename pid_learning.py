import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# System Parameters
# -------------------------------
m = 1.0     # mass (kg)
k = 20.0    # spring constant (N/m)
c = 2.0     # damping coefficient (N·s/m)

# -------------------------------
# Simulation Parameters
# -------------------------------
dt = 0.001
t_end = 5.0
time = np.arange(0, t_end, dt)

# -------------------------------
# Desired Position
# -------------------------------
x_target = 1.0

# -------------------------------
# PID Simulation Function
# -------------------------------
def simulate_pid(Kp, Ki, Kd, plot=False):
    x = 0.0
    v = 0.0
    integral = 0.0
    prev_error = 0.0

    x_history = []

    for t in time:
        error = x_target - x
        integral += error * dt
        derivative = (error - prev_error) / dt

        u = Kp * error + Ki * integral + Kd * derivative

        a = (u - c * v - k * x) / m
        v += a * dt
        x += v * dt

        prev_error = error
        x_history.append(x)

    if plot:
        plt.figure()
        plt.plot(time, x_history, label="Position")
        plt.axhline(x_target, linestyle="--", label="Target")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.title("PID Controlled Mass-Spring-Damper")
        plt.legend()
        plt.grid()

    return np.mean(np.abs(x_target - np.array(x_history)))

# -------------------------------
# Learning-Based PID Auto-Tuning
# -------------------------------
Kp, Ki, Kd = 50.0, 5.0, 10.0
learning_rate = 5.0
history = []

for i in range(30):
    error = simulate_pid(Kp, Ki, Kd)
    history.append(error)

    Kp -= learning_rate * (np.random.rand() - 0.5)
    Ki -= learning_rate * (np.random.rand() - 0.5)
    Kd -= learning_rate * (np.random.rand() - 0.5)

# -------------------------------
# Results
# -------------------------------
print("Learned PID Gains:")
print(f"Kp = {Kp:.2f}, Ki = {Ki:.2f}, Kd = {Kd:.2f}")

simulate_pid(Kp, Ki, Kd, plot=True)

plt.figure()
plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("Average Error")
plt.title("PID Learning Progress")
plt.grid()
plt.show()
