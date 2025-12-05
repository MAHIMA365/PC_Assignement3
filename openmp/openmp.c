%%bash
cat > omp_mergesort.c <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Function that merges two sorted halves of the array
void merge(int arr[], int left, int mid, int right) {
    // Sizes of the two temporary arrays
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Allocate temporary arrays
    int *L = malloc(n1 * sizeof(int));
    int *R = malloc(n2 * sizeof(int));

    // Copy data to temp arrays L[] and R[]
    for(int i = 0; i < n1; i++) L[i] = arr[left + i];
    for(int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    // Merge the two arrays back into arr[]
    int i = 0, j = 0, k = left;
    while(i < n1 && j < n2) {
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    }

    // Copy any leftover elements
    while(i < n1) arr[k++] = L[i++];
    while(j < n2) arr[k++] = R[j++];

    // Free temporary arrays
    free(L);
    free(R);
}

// Recursive merge sort using OpenMP tasks
void mergeSort(int arr[], int left, int right) {
    if(left >= right) return;

    int mid = (left + right) / 2;

    // Create task for sorting left half
    #pragma omp task shared(arr)
    mergeSort(arr, left, mid);

    // Create task for sorting right half
    #pragma omp task shared(arr)
    mergeSort(arr, mid + 1, right);

    // Wait until both tasks finish before merging
    #pragma omp taskwait
    merge(arr, left, mid, right);
}

int main(int argc, char *argv[]) {

    if(argc < 2) {
        printf("Usage: %s <num_threads>\n", argv[0]);
        return 0;
    }

    int threads = atoi(argv[1]);
    int n = 200000;   // size of the array to sort

    // Allocate memory for the array
    int *arr = (int *)malloc(n * sizeof(int));

    // Generate random numbers
    for(int i = 0; i < n; i++) arr[i] = rand();

    // Set number of threads for OpenMP
    omp_set_num_threads(threads);

    double start = omp_get_wtime();

    // Start parallel region (single thread will spawn tasks)
    #pragma omp parallel
    {
        #pragma omp single
        mergeSort(arr, 0, n - 1);
    }

    double end = omp_get_wtime();

    // Print execution time for the given thread count
    printf("Threads: %d | Time: %f seconds\n", threads, end - start);

    free(arr);
    return 0;
}
