/*
 * example calc of variance
 * -- var calcs inline
 *
 * Requirements
 * 1. file contains n, followed by the n datapoints
 *
 * mkbane (Nov 2024)
 * Modified by mnacler (Mar 2025)
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>  // for MPI

// function prototypes
int get_num_data_points(FILE*);
int read_data(FILE*, int, double*);  // returns number of points successfully read, populates array with data points

// main routine reading args from command line
int main(int argc, char** argv) {
    int rank, size, n, totalNum;
    double *x = NULL, *local_x = NULL, *squaredDiffs = NULL;
    int local_n;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Time total code (and elements thereof)
    double startTotalCode = MPI_Wtime();

    if (rank == 0) {
        // Master process reads the data
        FILE* filePtr;
        char *filename = argv[1]; // filename is 1st parameter on command line
        filePtr = fopen(filename, "r"); // open file, given by sole parameter, as read-only
        if (filePtr == NULL) {
            printf("Cannot open file %s\n", filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        totalNum = get_num_data_points(filePtr);
        printf("There are allegedly %d data points to read\n", totalNum);
        
        x = (double *) malloc(totalNum * sizeof(double));
        if (x == NULL) {
            printf("Error in allocating memory for data points\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        double start_readData = MPI_Wtime();
        n = read_data(filePtr, totalNum, x); // this is actual number of points read
        printf("%d data points successfully read [%f seconds]\n", n, MPI_Wtime()-start_readData);
        if (n != totalNum) printf("*** WARNING ***\n actual number read (%d) differs from header value (%d)\n\n", n, totalNum);
        
        if (n % size != 0) {
            printf("Number of data points must be divisible by number of processes\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast the number of data points to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_n = n / size;

    // Allocate memory for local data
    local_x = (double *) malloc(local_n * sizeof(double));
    if (local_x == NULL) {
        printf("Error in allocating memory for local data points\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Scatter the data to all processes
    MPI_Scatter(x, local_n, MPI_DOUBLE, local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calculate local sum
    double local_sum = 0.0;
    for (int i = 0; i < local_n; i++) {
        local_sum += local_x[i];
    }

    // Reduce local sums to get the global sum
    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Calculate the mean
    double mean;
    if (rank == 0) {
        mean = global_sum / (double) n;
    }

    // Broadcast the mean to all processes
    MPI_Bcast(&mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calculate local squared differences
    squaredDiffs = (double *) malloc(local_n * sizeof(double));
    if (squaredDiffs == NULL) {
        printf("Error in allocating memory for squared differences\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < local_n; i++) {
        double val = (local_x[i] - mean);
        squaredDiffs[i] = val * val;
    }

    // Calculate local sum of squared differences
    local_sum = 0.0;
    for (int i = 0; i < local_n; i++) {
        local_sum += squaredDiffs[i];
    }

    // Reduce local sums of squared differences to get the global sum
    double global_sum_squaredDiffs;
    MPI_Reduce(&local_sum, &global_sum_squaredDiffs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Calculate and print the variance
    if (rank == 0) {
        double variance = global_sum_squaredDiffs / (double) n;
        printf("The variance is %f\n", variance);
    }

    // Clean up
    if (rank == 0) {
        free(x);
    }
    free(local_x);
    free(squaredDiffs);

    // Finalize MPI
    MPI_Finalize();

    if (rank == 0) {
        printf("Completed. [%f seconds]\n", MPI_Wtime()-startTotalCode);
    }

    return 0;
}