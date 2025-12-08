Running OpenMP Merge Sort in Google Colab

This project implements parallel Merge Sort using OpenMP tasks in C.
These instructions explain how to compile and run the program inside Google Colab.

📌 Step 1: Upload or Create the C File in Colab

Upload the file:

omp_mergesort.c


Or create it inside Colab using:

%%bash
cat > omp_mergesort.c << 'EOF'
# (your code here)
EOF

📌 Step 2: Install GCC with OpenMP Support

Google Colab does not include the GCC compiler with OpenMP support by default.
Install it using:

%%bash
apt-get update
apt-get install -y gcc


Verify installation:

%%bash
gcc --version

📌 Step 3: Compile the Program

Use this command to compile your program with OpenMP support:

%%bash
gcc omp_mergesort.c -fopenmp -o omp_mergesort


This creates an executable named:

omp_mergesort

📌 Step 4: Run the Program

You must pass the number of threads as an argument.

Run with 1 thread:
%%bash
./omp_mergesort 1

Run with 4 threads:
%%bash
./omp_mergesort 4

Run with 8 threads:
%%bash
./omp_mergesort 8

📌 Example Output
Threads: 4 | Time: 0.213452 seconds

📌 Notes for Google Colab

Colab normally has 2 to 4 CPU cores, so using more than 4 threads may not improve performance.

Repeat the execution 3–5 times to get accurate timing.

If the Colab runtime resets, you must recompile the program.

Parallel Merge Sort using MPI

This program implements a parallel merge sort algorithm using MPI (Message Passing Interface).
The input array is divided equally among processes, each process sorts its local chunk, and the root process (rank 0) merges all sorted chunks to produce the final sorted array.

Features

Uses MPI_Scatter to distribute the array.

Each process performs a local merge sort in parallel.

Uses MPI_Gather to collect results.

Rank 0 performs the final merge.

Measures and prints execution time.

Compilation

Use mpicc to compile:

mpicc mpi_mergesort.c -o mpi_mergesort

Running the Program

Use mpirun or mpiexec:

mpirun -np <num_processes> ./mpi_mergesort <array_size>

Example
mpirun -np 4 ./mpi_mergesort 1000000

Program Flow

Rank 0 generates a random array of size n.

MPI broadcasts the size to all processes.

MPI scatters equal chunks to each process.

Each process sorts its chunk using merge sort.

Sorted chunks are gathered at rank 0.

Rank 0 merges all chunks into the final sorted array.

Execution time is printed.

Example (small array)

For array:
[38, 12, 7, 55, 4, 29, 13, 90] with 4 processes → chunks:

P0: [38,12] → [12,38]
P1: [7,55]  → [7,55]
P2: [4,29]  → [4,29]
P3: [13,90] → [13,90]


Final sorted array after merging:
[4, 7, 12, 13, 29, 38, 55, 90]

Tested On

GCC + MPICH/OpenMPI

Linux environments (including Google Colab)


