import matplotlib.pyplot as plt

processes = [1, 2, 4, 8, 16]

# Replace these with your actual measured times (milliseconds or seconds)
execution_times = [
    0.028580,  # T1
    0.026157,  # T2
    0.015390,  # T4
    0.013163,  # T8
    0.025349   # T16
]

# ===============================
# SPEEDUP CALCULATION
# ===============================
T1 = execution_times[0]
speedup = [T1 / t if t != 0 else 0 for t in execution_times]


# ===============================
# GRAPH 1: Processes vs Execution Time
# ===============================
plt.figure(figsize=(8,5))
plt.plot(processes, execution_times, marker='o')
plt.title("Number of Processes vs Execution Time")
plt.xlabel("Number of Processes")
plt.ylabel("Execution Time")
plt.grid(True)
plt.show()


# ===============================
# GRAPH 2: Processes vs Speedup
# ===============================
plt.figure(figsize=(8,5))
plt.plot(processes, speedup, marker='o')
plt.title("Number of Processes vs Speedup")
plt.xlabel("Number of Processes")
plt.ylabel("Speedup")
plt.grid(True)
plt.show()
