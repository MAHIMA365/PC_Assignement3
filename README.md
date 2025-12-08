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
