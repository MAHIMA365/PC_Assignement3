#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * merge()
 * Merges two sorted arrays 'a' and 'b' into output array 'c'.
 * a = left array of size n
 * b = right array of size m
 * c = merged sorted output of size n + m
 */
void merge(int *a, int n, int *b, int m, int *c) {
    int i = 0, j = 0, k = 0;

    // Compare both arrays and insert the smallest element first
    while(i < n && j < m)
        c[k++] = (a[i] <= b[j]) ? a[i++] : b[j++];

    // Copy remaining elements from left array if any
    while(i < n) c[k++] = a[i++];

    // Copy remaining elements from right array if any
    while(j < m) c[k++] = b[j++];
}

/*
 * merge_sort()
 * Standard recursive merge sort.
 * Splits the array into halves, sorts each half, and merges them.
 */
void merge_sort(int *a, int n) {
    // Base case: array of size <=1 is already sorted
    if(n < 2) return;

    int mid = n / 2;

    // Allocate memory for left and right halves
    int *left = malloc(mid * sizeof(int));
    int *right = malloc((n - mid) * sizeof(int));

    // Copy values into left and right arrays
    for(int i = 0; i < mid; i++) left[i] = a[i];
    for(int i = mid; i < n; i++) right[i - mid] = a[i];

    // Recursively sort both halves
    merge_sort(left, mid);
    merge_sort(right, n - mid);

    // Merge sorted halves back into original array
    merge(left, mid, right, n - mid, a);

    // Free temporary arrays
    free(left);
    free(right);
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);  // Initialize MPI environment

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

    // Ensure the user passed an array size argument
    if(rank == 0 && argc < 2) {
        printf("Usage: mpirun -np <processes> ./mpi_mergesort <n>\n");
        MPI_Finalize();
        return 0;
    }

    // Only rank 0 reads command-line argument; others set n=0 initially
    int n = (rank == 0) ? atoi(argv[1]) : 0;

    double start_time, end_time;

    // Broadcast the size of the array to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine size of each chunk to be handled by each process
    int chunk = n / size;

    int *full_array = NULL;

    // Rank 0 allocates and initializes the full array
    if(rank == 0) {
        full_array = malloc(n * sizeof(int));
        for(int i = 0; i < n; i++)
            full_array[i] = rand();   // Fill with random values
    }

    // Each process allocates memory for its sub-array
    int *sub_arr = malloc(chunk * sizeof(int));

    // Scatter equal-sized chunks of the array to all processes
    MPI_Scatter(full_array, chunk, MPI_INT,
                sub_arr, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    // Start timing AFTER scatter and before local sorting
    start_time = MPI_Wtime();

    // Each process sorts its sub-array locally
    merge_sort(sub_arr, chunk);

    int *gathered = NULL;

    // Rank 0 allocates memory to gather all sorted chunks
    if(rank == 0)
        gathered = malloc(n * sizeof(int));

    // Gather the sorted sub-arrays back to rank 0
    MPI_Gather(sub_arr, chunk, MPI_INT,
               gathered, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    // Rank 0 performs the final merge of all sorted chunks
    if(rank == 0) {

        int *tmp = malloc(n * sizeof(int));
        int size2 = chunk;  // Tracks how many elements are currently merged

        // Merge results one chunk at a time
        for(int p = 1; p < size; p++) {
            merge(gathered, size2, gathered + p * chunk, chunk, tmp);

            // Copy merged result back into gathered[]
            for(int i = 0; i < size2 + chunk; i++)
                gathered[i] = tmp[i];

            size2 += chunk;  // Increase merged size
        }

        // Stop timer after final merge
        end_time = MPI_Wtime();

        // Print performance result
        printf("Processes: %d | Time: %f seconds\n", size, end_time - start_time);

        free(tmp);
        free(gathered);
        free(full_array);
    }

    // Free the sub-array in each process
    free(sub_arr);

    MPI_Finalize(); // End MPI environment
    return 0;
}

