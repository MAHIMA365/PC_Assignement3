📌 Step 1: Upload the C File to Colab
%%bash
apt-get update
apt-get install -y gcc
To verify
%%bash
gcc --version
📌 Step 3: Compile the Program
%%bash
gcc omp_mergesort.c -fopenmp -o omp_mergesort
omp_mergesort
📌 Step 4: Run the Program
Run with 1 thread:
%%bash
./omp_mergesort 1
......

