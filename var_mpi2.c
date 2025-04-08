/*
 * example calc of variance
 * -- portable distributed memory implementation using MPI
 *
 * Requirements
 * 1. file contains n, followed by the n datapoints
 *
 * mkbane (Nov 2024), modified by Copilot (Apr 2025)
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// function prototypes
int get_num_data_points(FILE*);
int read_data(FILE*, int, double*);  // returns number of points successfully read, populates array with data points

// main routine reading args from command line
int main(int argc, char** argv) {
    int n;                  // number of data points
    double *x;              // pointer to array holding data points
    double *squaredDiffs;   // pointer to array holding squared differences (of x from mean of all x)
    int rank, size;         // rank of the process and number of processes

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // time total code (and elements thereof)
    double startTotalCode = MPI_Wtime();

    if (rank == 0) {
        // access file here (and then pass pointer to file). This allows >1 routine to access same file.
        FILE* filePtr;
        char *filename = argv[1]; // filename is 1st parameter on command line
        filePtr = fopen(filename, "r"); // open file, given by sole parameter, as read-only
        if (filePtr == NULL) {
            printf("Cannot open file %s\n", filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        } else {
            int totalNum = get_num_data_points(filePtr);
            printf("There are allegedly %d data points to read\n", totalNum);
            x = (double *) malloc(totalNum * sizeof(double));
            if (x == NULL) {
                // error in allocating memory
                printf("Error in allocating memory for data points\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            } else {
                double start_readData = MPI_Wtime();
                n = read_data(filePtr, totalNum, x); // this is actual number of points read
                printf("%d data points successfully read [%f seconds]\n", n, MPI_Wtime() - start_readData);
                if (n != totalNum) printf("*** WARNING ***\n actual number read (%d) differs from header value (%d)\n\n", n, totalNum);
            }
        }
    }

    // Broadcast n to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        x = (double *) malloc(n * sizeof(double));
    }

    // Broadcast the data points to all processes
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    squaredDiffs = (double *) malloc(n * sizeof(double));
    if (squaredDiffs == NULL) {
        // error in allocating memory
        printf("Error in allocating memory for squared differences\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /*
     * main data processing loop
     *
     */

    int myNum = n/size;
    int myStart = rank*myNum;

    if (rank == size-1) {
	myNum = n - (size-1)*myNum;
    }

    int myFinish = myStart + myNum;

//    printf("rank %f\n", rank);
//    printf("size %f\n", size);

    if (rank == 0) {
	printf("x[0]=%f\n", x[0]);
    }

    double local_sum = 0.0;
    double mean;
    double start = MPI_Wtime();

    double local_minabs = fabs(x[rank]);
    double local_maxabs = fabs(x[rank]);

    // Calculate the sum of x values
    for (int i = myStart; i < myFinish; i++) {
        local_sum += x[i];
	if (fabs(x[i]) < local_minabs) local_minabs = fabs(x[i]);
	if (fabs(x[i]) > local_maxabs) local_maxabs = fabs(x[i]);
    }

    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        mean = global_sum / (double) n;
      //  printf("Sum of x values: %f\n", global_sum);
      //  printf("Mean: %f\n", mean);
    }

    // Broadcast the mean to all processes
    MPI_Bcast(&mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    local_sum = 0.0;
    // Calculate squared differences
    for (int i = myStart; i < myFinish; i++) {
        //double val = (x[i] - mean);
        squaredDiffs[i] = (x[i] - mean)*(x[i] - mean); //val * val;
	local_sum += squaredDiffs[i];
    }

    //local_sum = 0.0;

    // Calculate the sum of squared differences
    //for (int i = rank; i < n; i += size) {
    //    local_sum += squaredDiffs[i];
    //}

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    //if (rank == 0) {
    //    double variance = global_sum / (double) n;
    //    printf("Sum of squared differences: %f\n", global_sum);
    //    printf("Variance: %f\n", variance);
    //}

    // Find minimum and maximum absolute values
    //double local_minabs = fabs(x[rank]);
    //double local_maxabs = fabs(x[rank]);

    //for (int i = rank; i < n; i += size) {
        //double val = fabs(x[i]);
    //    if (fabs(x[i]) < local_minabs) local_minabs = fabs(x[i]); //val;
    //    if (fabs(x[i]) > local_maxabs) local_maxabs = fabs(x[i]); //val;
    //}

    double global_minabs, global_maxabs;
    MPI_Reduce(&local_minabs, &global_minabs, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_maxabs, &global_maxabs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
	printf("Total parallel time [%f seconds]\n", MPI_Wtime() - start);
	double variance = global_sum / (double) n;
	printf("Mean: %f\n", mean);
	printf("Variance: %f\n", variance);
        printf("Min absolute value: %f\n", global_minabs);
        printf("Max absolute value: %f\n", global_maxabs);
    }

    // Free allocated memory
    free(x);
    free(squaredDiffs);

    //MPI_Finalize();

    if (rank == 0) {
        printf("Completed. [%f seconds]\n", MPI_Wtime() - startTotalCode);
    }

    return 0;

    MPI_Finalize();
}
