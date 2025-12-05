import matplotlib.pyplot as plt

threads = [1, 2, 4, 8, 16]
times = [0.112192, 0.091469, 0.108268, 0.109398, 0.204486]  # <-- put measured times here

# Calculate speedup
speedup = [times[0] / t for t in times]

# --- Execution Time Graph ---
plt.figure(figsize=(6,4))
plt.plot(threads, times, marker='o')
plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (seconds)")
plt.title("Threads vs Execution Time")
plt.grid(True)
plt.show()

# --- Speedup Graph ---
plt.figure(figsize=(6,4))
plt.plot(threads, speedup, marker='o')
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Threads vs Speedup")
plt.grid(True)
plt.show()
