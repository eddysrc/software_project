/*
A C program that implements K-Means algorithm.
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>

int snprintf(char *__stream, size_t __n, const char *__format, ...);

typedef struct {
    int dimension;
    double *centroid;
    double *vectorsSum;
    int closestVectorsCounter;
}Cluster;

void safeMalloc(void * pointer){
    if (pointer == NULL){
        printf("An Error Has Occurred\n");
        exit(1);
    }
}

void determineSizeMatrix(char* inputFileName, int* sizesArray){
    int numOfVectors = 0;
    int dimension = 0;
    int maxWordSize = 0;
    int currentWordSize = 0;
    char currentChar = 'a';
    
    FILE *ifp = NULL;
    ifp = fopen(inputFileName, "r");
    
    while (currentChar != EOF){
        currentChar = fgetc(ifp);
        if (currentChar == '\n'){
            numOfVectors += 1;
            if (currentWordSize > maxWordSize){
                maxWordSize = currentChar;
            }
            currentWordSize = 0;
        }
        else if (currentChar == ','){
            dimension += 1;
            if (currentWordSize > maxWordSize){
                maxWordSize = currentChar;
            }
            currentWordSize = 0;
        }
        else{
            currentWordSize += 1;
        }
    }
    dimension += numOfVectors;
    dimension /= (int) numOfVectors;

    sizesArray[0] = numOfVectors;
    sizesArray[1] = dimension;
    sizesArray[2] = maxWordSize;

    fclose(ifp);
}

void loadMatrix(double** matrix, int rows, int columns, int maxWordSize, char *inputFileName){
    int i, j;
    FILE *ifp = NULL;
    ifp = fopen(inputFileName, "r");

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < columns; j++){
            char *currentWord = (char*) malloc(sizeof(char) * (maxWordSize));
            char currentChar = fgetc(ifp);
            int currentCharIndex = 0;
            safeMalloc(currentWord);

            while (currentChar != '\n' && currentChar != EOF && currentChar != ','){
                currentWord[currentCharIndex] = currentChar;
                currentCharIndex += 1;
                currentChar = fgetc(ifp);
            }
            
            if (currentCharIndex < maxWordSize){
                currentWord[currentCharIndex] = '\0';
            }

            matrix[i][j] = atof(currentWord);

            free(currentWord);
        }
    }    
    fclose(ifp);
}

void setCentroid(Cluster cluster, double *vector, int dimension){
    int i;

    for (i = 0; i < dimension; i++){
        cluster.centroid[i] = vector[i];
    }
}


/* Open file and initialize cluster centroids with first k lines. */
void initializeClusters(Cluster *clusterList, double **matrix, int k, int dimension){
    int i, j;

    for (i = 0; i < k; i++){
        /* New cluster */
        Cluster newCluster;
        double *newCentroid = (double *) malloc(sizeof(double) * dimension);
        double *vectorsSum = (double *) malloc(sizeof(double) * dimension);
        safeMalloc(newCentroid);
        safeMalloc(vectorsSum);
        for (j = 0; j < dimension; j++){
            vectorsSum[j] = 0.0;
        }
        newCluster.dimension = dimension;
        newCluster.centroid = newCentroid;
        newCluster.vectorsSum = vectorsSum;
        newCluster.closestVectorsCounter = 0;
        setCentroid(newCluster, matrix[i], dimension);
        clusterList[i] = newCluster;
    }
}

void subtractVectors(double* vector1, double* vector2, int dim, double* subVector){
    int i;

    for (i = 0; i < dim; i++)
    {
        subVector[i] = vector1[i] - vector2[i];
    }
}

void addVectors(double* vector1, double* vector2, int dim, double* addVector){
    int i;

    for (i = 0; i < dim; i++)
    {
        addVector[i] = vector1[i] + vector2[i];
    }
}

double calculateDistanceFromCentroid(double *vector1, double *vector2, int dim){
    int i;
    double *subtractedVector = (double*) malloc(sizeof(double) * dim);
    double distance = 0;
    safeMalloc(subtractedVector);

    subtractVectors(vector1, vector2, dim, subtractedVector);
    
    for (i = 0; i < dim; i++)
    {
        distance += pow(subtractedVector[i], 2);
    }
    
    free(subtractedVector);

    return distance;
}

/* Determine closest Cluster */
int findClosestCluster(double *vector, Cluster *clusterList, int k){
    int i;
    int minIndex = 0;

    double minDistance = calculateDistanceFromCentroid(vector, clusterList[0].centroid, clusterList[0].dimension);
    for (i = 0; i < k; i++)
    {
        double currentDistance = calculateDistanceFromCentroid(vector, clusterList[i].centroid, clusterList[i].dimension);
        if (currentDistance < minDistance){
            minDistance = currentDistance;
            minIndex = i;
        }   
    }
    return minIndex;
}

void updateClusters(double **matrix, Cluster *clusterList, int rows, int k){
    int i, j, m;

    for (i = 0; i < rows; i++)
    {
        Cluster *closestCluster = &clusterList[findClosestCluster(matrix[i], clusterList, k)];
        addVectors((*closestCluster).vectorsSum, matrix[i], (*closestCluster).dimension, (*closestCluster).vectorsSum);
        (*closestCluster).closestVectorsCounter += 1;
    }

    for (j = 0; j < k; j++)
    {
        for (m = 0; m < clusterList[0].dimension; m++)
        {
            if (clusterList[j].closestVectorsCounter != 0){
                clusterList[j].centroid[m] = clusterList[j].vectorsSum[m] / clusterList[j].closestVectorsCounter;
            }
            clusterList[j].vectorsSum[m] = 0.0;
        }
        clusterList[j].closestVectorsCounter = 0;
    }    
}

double calcEuclideanNorm(double *vector1, double *vector2, int dim){
    int i;
    double sum = 0.0;
    double *subVector = (double*) malloc(sizeof(double) * dim);
    safeMalloc(subVector);

    subtractVectors(vector1, vector2, dim, subVector);
    
    for (i = 0; i < dim; i++)
    {
        sum += pow(subVector[i], 2);
    }

    free(subVector);

    return sqrt(sum);
}


void calculateEuclideanNorms(double *deltaVectors, double **beforeCentroid, double ** afterCentroid, int k, int dimension){
    int i;

    for (i = 0; i < k; i++)
    {
        deltaVectors[i] = calcEuclideanNorm(beforeCentroid[i], afterCentroid[i], dimension);
    }
}


int checkEuclideanNorms(double *deltaVectors, double epsilon, int k){
    int i;

    for (i = 0; i < k; i++)
    {
        if (deltaVectors[i] > epsilon){
            return 1;
        }
    }
    return 0;
}


void writeToOutput(char *outputFileName, Cluster *clusterList, int k, int dim){
    int i, j;
    FILE *ifp = NULL;
    ifp = fopen(outputFileName, "w");

    for (i = 0; i < k; i++)
    {
        for (j = 0; j < dim; j++)
        {
            /* Initiate empty string for float. */
            char outputStr[24];
            snprintf(outputStr, 24, "%.4f",clusterList[i].centroid[j]);
            /* Write data to file */
            fprintf(ifp, "%s", outputStr);

            /* Add special chars to output file. */
            if (j == dim - 1){
                fputc('\n', ifp);
            }
            else {
                fputc(',', ifp);
            }
        }
        
    }
    fclose(ifp);
}

int main(int argc, char *argv[]){
    int i, j;
    int currentIter, rows, columns, maxWordLength;
    int dimension;

    int k, maxIter;
    char *inputFileName;
    char *outputFileName;
    double const EPSILON = 0.001;

    Cluster *clusterList;
    double **matrix;

    int sizesArray[3];
    double *deltaVectors;

    char *maxIterPointer;

    if (argc == 5){ /* Case max_iter is provided */
        char *kPointer = argv[1];
        while (*argv[1] && isdigit(*argv[1])){
            argv[1] += 1;
        }
        if (*argv[1] != '\0'){
            printf("Invalid Input!\n");
            exit(1);
        }

        maxIterPointer = argv[2];
        while (*argv[2] && isdigit(*argv[2])){
            argv[2] += 1;
        }
        if (*argv[2] != '\0'){
            printf("Invalid Input!\n");
            exit(1);
        }

        /* Need to handle casting! */
        k = atoi(kPointer);
        maxIter = atoi(maxIterPointer);
        inputFileName = argv[3];
        outputFileName = argv[4];
    }
    else if (argc == 4){ /* Case max_iter is not provided */
        char *kPointer = argv[1];
        while (*argv[1] && isdigit(*argv[1])){
            argv[1] += 1;
        }
        if (*argv[1] != '\0'){
            printf("Invalid Input!\n");
            exit(1);
        }
        k = atoi(kPointer);
        maxIter = 200; /* Default value */
        inputFileName = argv[2];
        outputFileName = argv[3];
    }
    else {
        /* Too many or too few arguments */
        printf("Invalid Input!\n");
        exit(1);
    }

    currentIter = 0;
    clusterList = (Cluster*) malloc(sizeof(Cluster) * k);
    safeMalloc(clusterList);
    determineSizeMatrix(inputFileName, sizesArray);
    rows = sizesArray[0];
    columns = sizesArray[1];
    maxWordLength = sizesArray[2];

    if (rows < k){
        printf("Invalid Input!\n");
        exit(1);
    }

    matrix=(double**)malloc(rows * sizeof(double *));
    safeMalloc(matrix);

    for(i = 0; i < rows; i++)
    {
        matrix[i] = (double*) malloc(columns * sizeof(double));
        safeMalloc(matrix[i]);
    }

    loadMatrix(matrix, rows, columns, maxWordLength, inputFileName);

    initializeClusters(clusterList, matrix, k, columns);
    
    dimension = clusterList[0].dimension;
    
    /* Initialize euclidian norms deltas. */
    deltaVectors = (double *) malloc(sizeof(double) * k);
    safeMalloc(deltaVectors);

    for (i = 0; i < k; i++){
        deltaVectors[i] = 1.0;
    }

    while (checkEuclideanNorms(deltaVectors, EPSILON, dimension) && currentIter < maxIter)
    {
        double **centroidsBefore=(double**)malloc(k * sizeof(double *));
        double **centroidsAfter=(double**)malloc(k * sizeof(double *));
        safeMalloc(centroidsBefore);
        safeMalloc(centroidsAfter);

        for(i = 0; i < k; i++)
        {
            centroidsBefore[i] = (double*) malloc(dimension * sizeof(double));
            safeMalloc(centroidsBefore[i]);
        }

        for (i = 0; i < k; i++)
        {
            for (j = 0; j < dimension; j++)
            {
                centroidsBefore[i][j] = clusterList[i].centroid[j];
            }
        }

        updateClusters(matrix, clusterList, rows, k);
        
        for(i = 0; i < k; i++)
        {
            centroidsAfter[i] = (double*) malloc(dimension * sizeof(double));
            safeMalloc(centroidsAfter[i]);
        }

        for (i = 0; i < k; i++)
        {
            for (j = 0; j < dimension; j++)
            {
                centroidsAfter[i][j] = clusterList[i].centroid[j];
            }
        }

        calculateEuclideanNorms(deltaVectors, centroidsBefore, centroidsAfter, k, dimension);
        currentIter += 1;

        /* Free used space for centroids */ 
        for (i = 0; i < k; i++)
        {
            free(centroidsBefore[i]);
            free(centroidsAfter[i]);
        }
        free(centroidsBefore);
        free(centroidsAfter);
    }

    writeToOutput(outputFileName, clusterList, k, dimension);
    
    /* Free used space for data structures */
    for (i = 0; i < rows; i++)
    {
        free(matrix[i]);
    }
    free(matrix);

    for (i = 0; i < k; i++){
        free(clusterList[i].centroid);
        free(clusterList[i].vectorsSum);
    }

    free(deltaVectors);
    free(clusterList);

    return 0;
}
