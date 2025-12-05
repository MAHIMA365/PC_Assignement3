# create cuda_mergesort.cu
%%bash
cat > cuda_mergesort.cu <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

__global__ void merge_kernel(int* d_arr, int width, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * (2 * width);
    if (start >= n) return;

    int mid = start + width;
    int end = start + 2 * width;

    if (mid > n) mid = n;
    if (end > n) end = n;

    int i = start;
    int j = mid;
    int k = 0;

    int* temp = (int*)malloc((end - start) * sizeof(int));

    while (i < mid && j < end) {
        if (d_arr[i] < d_arr[j]) temp[k++] = d_arr[i++];
        else temp[k++] = d_arr[j++];
    }

    while (i < mid) temp[k++] = d_arr[i++];
    while (j < end) temp[k++] = d_arr[j++];

    for (int x = 0; x < k; x++)
        d_arr[start + x] = temp[x];

    free(temp);
}

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void print_array(int* arr, int n)
{
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        printf("Usage: ./cuda_mergesort <array_size> <block_size> <threads>\n");
        return 1;
    }

    int n = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    int threads = atoi(argv[3]);

    int* h_arr = (int*)malloc(n * sizeof(int));

    // random array values
    for (int i = 0; i < n; i++)
        h_arr[i] = rand() % 10000;

    int* d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    double start_time = get_time();

    for (int width = 1; width < n; width *= 2) {
        int blocks = block_size;
        merge_kernel<<<blocks, threads>>>(d_arr, width, n);
        cudaDeviceSynchronize();
    }

    double end_time = get_time();

    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Array Size: %d | Blocks: %d | Threads: %d | Time: %.6f sec\n",
        n, block_size, threads, end_time - start_time);

    cudaFree(d_arr);
    free(h_arr);

    return 0;
}
