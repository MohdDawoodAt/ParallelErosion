#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 890 // number of rows
#define M 750 // number of columns

__global__ void erosion_kernel(int (*image)[M], int (*result)[M])
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row > 0 && row < N - 1 && col > 0 && col < M - 1)
    {
        result[row][col] = image[row][col] & image[row - 1][col] & image[row + 1][col] & image[row][col - 1] &
                           image[row][col + 1] & image[row + 1][col + 1] & image[row - 1][col + 1] &
                           image[row + 1][col - 1] & image[row - 1][col - 1];
    }
    else
    {
        result[row][col] = image[row][col];
    }
}

int main()
{
    int(*binaryImage)[M], (*result)[M];
    int i, j;

    // Allocate matrices on unified memory
    cudaMallocManaged(&binaryImage, N * sizeof(int[M]));
    cudaMallocManaged(&result, N * sizeof(int[M]));

    // Fill the binaryImage matrix with random binary values
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            binaryImage[i][j] = rand() % 2;
        }
    }

    dim3 grid_dims, block_dims;
    grid_dims.x = 28; // adjust the grid size based on your image dimensions
    grid_dims.y = 30; // adjust the grid size based on your image dimensions
    block_dims.x = N / grid_dims.x;
    block_dims.y = M / grid_dims.y;

    // Call the erosion kernel
    erosion_kernel<<<grid_dims, block_dims>>>(binaryImage, result);
    cudaDeviceSynchronize();

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

    // Free allocated memory
    cudaFree(binaryImage);
    cudaFree(result);

    return 0;
}
