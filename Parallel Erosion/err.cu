#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 200 // number of rows
#define M 200 // number of columns

__global__ void erosion_kernel(int **image, int **result, int *flattenedKernel, int x, int y)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int cell = 1;
    int xthresh = x - x / 2 - 1;
    int ythresh = y - y / 2 - 1;

    if (row > xthresh && row < N - xthresh && col > ythresh && col < M - ythresh)
    {

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                if (flattenedKernel[i * y + j])
                    cell = cell && (flattenedKernel[i * y + j] & image[row - 1 + j][col - 1 + i]);
            }
        }

        result[row][col] = cell;
    }
    else
    {
        result[row][col] = 0;
    }
}

int main()
{
    int **binaryImage, **result;
    int *flattenedKernel;
    int i, j, x = 5, y = 5; // dimensions of the kernel
    double cpu_time;
    clock_t begin, end;

    // Allocate matrices on unified memory
    cudaMallocManaged(&binaryImage, N * sizeof(int *));
    cudaMallocManaged(&result, N * sizeof(int *));
    cudaMallocManaged(&flattenedKernel, x * y * sizeof(int));

    for (i = 0; i < N; i++)
    {
        cudaMallocManaged(&binaryImage[i], M * sizeof(int));
        cudaMallocManaged(&result[i], M * sizeof(int));
    }

    FILE *file = fopen("200.txt", "r");
    if (file == NULL)
    {
        perror("Error opening file");
        return 1;
    }

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            if (fscanf(file, "%d", &binaryImage[i][j]) == EOF)
            {
                if (feof(file))
                {
                    fprintf(stderr, "End of file reached unexpectedly\n");
                }
                else if (ferror(file))
                {
                    perror("Error reading from file");
                }
                else
                {
                    fprintf(stderr, "Unexpected error reading from file\n");
                }
                fclose(file);
                return 1;
            }
        }
    }

    fclose(file);

    // Fill the kernel matrix with desired values
    int krnel[x][y] = {
        {0, 0, 1, 0, 0},
        {0, 1, 1, 1, 0},
        {1, 1, 1, 1, 1},
        {0, 1, 1, 1, 0},
        {0, 0, 1, 0, 0}};

    // Flatten the kernel matrix
    for (i = 0; i < x; i++)
    {
        for (j = 0; j < y; j++)
        {
            flattenedKernel[i * y + j] = krnel[i][j];
        }
    }

    dim3 grid_dims, block_dims;
    grid_dims.x = N;
    grid_dims.y = M;
    block_dims.x = N / grid_dims.x;
    block_dims.y = M / grid_dims.y;

    begin = clock();
    // Call the erosion kernel
    erosion_kernel<<<grid_dims, block_dims>>>(binaryImage, result, flattenedKernel, x, y);
    cudaDeviceSynchronize();
    end = clock();

    FILE *outputFile = fopen("eroded_result200cu.txt", "w");
    if (outputFile == NULL)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            fprintf(outputFile, "%d ", result[i][j]);
        }
        fprintf(outputFile, "\n");
    }

    fclose(outputFile);

    cpu_time = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("cpu time = %.4f\n", cpu_time);

    // Free allocated memory
    cudaFree(binaryImage);
    cudaFree(result);
    cudaFree(flattenedKernel);

    return 0;
}
