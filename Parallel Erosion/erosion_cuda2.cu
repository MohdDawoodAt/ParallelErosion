#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 1000 // number of rows
#define M 1000 // number of columns

__global__ void erosion_kernel(int(**image), int(**result), int(**kernel), int x, int y)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int cell = 1;

    if (row > 0 && row < N - 1 && col > 0 && col < M - 1)
    {

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                if (kernel[i][j])
                    cell = cell & (kernel[i][j] & image[row - 1 + i][col - 1 + j]);
            }
        }
        result[row][col] = cell;

        //  image[row][col] & image[row - 1][col] & image[row + 1][col] & image[row][col - 1]  & image[row][col + 1];
        // & image[row + 1][col + 1] & image[row - 1][col + 1] &
        //                    image[row + 1][col - 1] & image[row - 1][col - 1];
    }
    else
    {
        result[row][col] = 0;
    }
}

int main()
{
    int **binaryImage, **result, **kernel;
    int i, j, x, y;
    double cpu_time;
    clock_t begin, end;

    // Allocate matrices on unified memory
    cudaMallocManaged(&binaryImage, N * sizeof(int *));
    cudaMallocManaged(&result, N * sizeof(int *));
    cudaMallocManaged(&kernel, N * sizeof(int *));

    for (i = 0; i < N; i++)
    {
        cudaMallocManaged(&binaryImage[i], M * sizeof(int));
        cudaMallocManaged(&result[i], M * sizeof(int));

        if (i < y)
            cudaMallocManaged(&kernel[i], M * sizeof(int));
    }

    // Fill the binaryImage matrix with random binary values
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            binaryImage[i][j] = rand() % 2;
        }
    }
    printf("\nBinary Image before Erosion:\n");
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            printf("%d ", binaryImage[i][j]);
        }
        printf("\n");
    }
    dim3 grid_dims, block_dims;
    grid_dims.x = N; // adjust the grid size based on your image dimensions
    grid_dims.y = M; // adjust the grid size based on your image dimensions
    block_dims.x = N / grid_dims.x;
    block_dims.y = M / grid_dims.y;

    begin = clock();
    // Call the erosion kernel
    kernel = {
        {0, 1, 0},
        {1, 1, 1},
        {0, 1, 0}};
    erosion_kernel<<<grid_dims, block_dims>>>(binaryImage, result, kernel, 3, 3);
    cudaDeviceSynchronize();
    end = clock();

    // Print the eroded binary image
    printf("\nBinary Image after Erosion:\n");
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            printf("%d ", result[i][j]);
        }
        printf("\n");
    }

    cpu_time = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("cpu time = %.4f\n", cpu_time);

    // Free allocated memory
    cudaFree(binaryImage);
    cudaFree(result);

    return 0;
}
