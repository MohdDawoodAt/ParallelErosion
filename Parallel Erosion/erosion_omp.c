#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define N 512 // number of rows
#define M 512 // number of columns

void erosion_kernel(int **image, int **result, int *flattenedKernel, int x, int y)
{
    int xthresh = x - x / 2 - 1;
    int ythresh = y - y / 2 - 1;
#pragma omp parallel for shared(image, result, flattenedKernel) num_threads(8)
    for (int row = xthresh; row < N - xthresh; row++)
    {
        for (int col = ythresh; col < M - ythresh; col++)
        {
            int cell = 1;

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
    }
}

int main()
{
    double cpu_time;
    double begin, end;
    int **binaryImage, **result;
    int *flattenedKernel;
    int i, j, x = 9, y = 9; // dimensions of the kernel

    // Allocate matrices
    binaryImage = (int **)malloc(N * sizeof(int *));
    result = (int **)malloc(N * sizeof(int *));
    flattenedKernel = (int *)malloc(x * y * sizeof(int));

    for (i = 0; i < N; i++)
    {
        binaryImage[i] = (int *)malloc(M * sizeof(int));
        result[i] = (int *)malloc(M * sizeof(int));
    }

    FILE *file = fopen("512.txt", "r");
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

    int krnel[9][9] = {
        {0, 0, 1, 1, 1, 1, 1, 0, 0},
        {0, 1, 1, 1, 1, 1, 1, 1, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1},
        {0, 1, 1, 1, 1, 1, 1, 1, 0},
        {0, 0, 1, 1, 1, 1, 1, 0, 0}};

    // Flatten the kernel matrix
    for (i = 0; i < x; i++)
    {
        for (j = 0; j < y; j++)
        {
            flattenedKernel[i * y + j] = krnel[i][j];
        }
    }

    begin = omp_get_wtime();
    // Call the erosion kernel
    erosion_kernel(binaryImage, result, flattenedKernel, x, y);
    end = omp_get_wtime();

    FILE *outputFile = fopen("eroded_result512omp.txt", "w");
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
        fprintf(outputFile, "l\n");
    }

    fclose(outputFile);

    cpu_time = end - begin;
    printf("cpu time = %.4f\n", cpu_time);

    // Free allocated memory
    for (i = 0; i < N; i++)
    {
        free(binaryImage[i]);
        free(result[i]);
    }
    free(binaryImage);
    free(result);
    free(flattenedKernel);

    return 0;
}
