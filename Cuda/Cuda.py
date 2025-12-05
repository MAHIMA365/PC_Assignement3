import matplotlib.pyplot as plt


configs = [
    "32B-32T",
    "32B-64T",
    "64B-64T",
    "128B-128T",
    "256B-256T"
]

# Replace these values with the execution time printed by your program
execution_times = [
    0.043355,   # Time for 32B-32T
    0.014909,   # Time for 32B-64T
    0.007353,   # Time for 64B-64T
    0.007423,   # Time for 128B-128T
    0.007228    # Time for 256B-256T
]

# =========================================================
# SPEEDUP CALCULATION
# =========================================================

# Baseline = first configuration
T_base = execution_times[0]

speedup = [T_base / t if t != 0 else 0 for t in execution_times]


# =========================================================
# GRAPH 1: CONFIGURATION vs EXECUTION TIME
# =========================================================
plt.figure(figsize=(9,5))
plt.plot(configs, execution_times, marker='o')
plt.title("CUDA Configuration vs Execution Time")
plt.xlabel("Configuration (Blocks-Threads)")
plt.ylabel("Execution Time (seconds)")
plt.grid(True)
plt.show()

# =========================================================
# GRAPH 2: CONFIGURATION vs SPEEDUP
# =========================================================
plt.figure(figsize=(9,5))
plt.plot(configs, speedup, marker='o')
plt.title("CUDA Configuration vs Speedup")
plt.xlabel("Configuration (Blocks-Threads)")
plt.ylabel("Speedup")
plt.grid(True)
plt.show()
