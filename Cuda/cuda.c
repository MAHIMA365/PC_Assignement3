#include <stdio.h>      // For printf
#include <stdlib.h>     // For malloc, rand, atoi
#include <cuda.h>       // CUDA library
#include <sys/time.h>   // For timing with gettimeofday

// -------------------------------
// GPU Kernel: performs merge of two sorted subarrays
// width = size of each subarray (1,2,4,8,...)
// d_arr = device array
// n = total elements
// -------------------------------
__global__ void merge_kernel(int* d_arr, int width, int n)
{
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread merges one pair of subarrays
    int start = tid * (2 * width);

    // If start index exceeds array size → exit
    if (start >= n) return;

    // Compute mid and end boundary of merge segment
    int mid = start + width;
    int end = start + 2 * width;

    // Limit mid and end to array bounds
    if (mid > n) mid = n;
    if (end > n) end = n;

    // Left half pointer
    int i = start;

    // Right half pointer
    int j = mid;

    // Temp array pointer index
    int k = 0;

    // Allocate temporary space INSIDE GPU kernel
    // (slow and not recommended for real GPU programs)
    int* temp = (int*)malloc((end - start) * sizeof(int));

    // Merge two sorted halves
    while (i < mid && j < end) {
        if (d_arr[i] < d_arr[j])
            temp[k++] = d_arr[i++];
        else
            temp[k++] = d_arr[j++];
    }

    // Copy remaining elements from left half
    while (i < mid)
        temp[k++] = d_arr[i++];

    // Copy remaining elements from right half
    while (j < end)
        temp[k++] = d_arr[j++];

    // Write merged results back to global memory
    for (int x = 0; x < k; x++)
        d_arr[start + x] = temp[x];

    // Free temporary buffer
    free(temp);
}


// Get time in seconds using gettimeofday()

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}


// Print an array (for debugging)

void print_array(int* arr, int n)
{
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");
}


// Main Function

int main(int argc, char** argv)
{
    // Expect: array size, block count, threads per block
    if (argc < 4) {
        printf("Usage: ./cuda_mergesort <array_size> <block_size> <threads>\n");
        return 1;
    }

    // Read input values
    int n = atoi(argv[1]);            // Total array size
    int block_size = atoi(argv[2]);   // Number of blocks
    int threads = atoi(argv[3]);      // Threads per block

    // Allocate host array
    int* h_arr = (int*)malloc(n * sizeof(int));

    // Fill with random values
    for (int i = 0; i < n; i++)
        h_arr[i] = rand() % 10000;

    // Allocate GPU memory
    int* d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));

    // Copy array from host → device
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Start timing
    double start_time = get_time();

    // Perform merge passes with widths: 1, 2, 4, 8, ...
    for (int width = 1; width < n; width *= 2) {

        // Launch kernel with user-defined blocks and threads
        int blocks = block_size;

        merge_kernel<<<blocks, threads>>>(d_arr, width, n);
        cudaDeviceSynchronize();     // Wait for GPU to finish
    }

    // Stop timing
    double end_time = get_time();

    // Copy sorted array back to host
    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print statistics
    printf("Array Size: %d | Blocks: %d | Threads: %d | Time: %.6f sec\n",
        n, block_size, threads, end_time - start_time);

    // Free GPU and CPU memory
    cudaFree(d_arr);
    free(h_arr);

    return 0;
}
