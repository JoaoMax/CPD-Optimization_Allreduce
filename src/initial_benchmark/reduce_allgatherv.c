#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CORES_PER_NODE 12
#define NUM_BARRIERS 4

int main(int argc, char **argv) {
    int mpi_size, mpi_rank;
    int *recvcounts, *displs;  // Para Allgatherv
    int sendcount, iterations, i;
    double wtime, wtime_sum;
    char *sendbuf, *recvbuf, *finalbuf;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // Configuración inicial
    sendcount = 1024; // Tamaño del mensaje
    iterations = 100;

    // Asignación de buffers
    sendbuf = (char *)malloc(sendcount * sizeof(char));
    recvbuf = (char *)malloc(sendcount * sizeof(char));
    finalbuf = (char *)malloc(sendcount * mpi_size * sizeof(char)); // Buffer final

    memset(sendbuf, 1, sendcount);
    memset(recvbuf, 0, sendcount);
    memset(finalbuf, 0, sendcount * mpi_size);

    // Configurar desplazamientos y tamaños para Allgatherv
    recvcounts = (int *)malloc(mpi_size * sizeof(int));
    displs = (int *)malloc(mpi_size * sizeof(int));
    for (int i = 0; i < mpi_size; i++) {
        recvcounts[i] = sendcount / mpi_size;  // División equitativa
        displs[i] = i * recvcounts[i];         // Desplazamientos
    }

    // Bucle principal con Reduce_scatter y Allgatherv
    wtime_sum = 0.0;
    for (i = 0; i < iterations; i++) {
        MPI_Barrier(MPI_COMM_WORLD);  // Sincronización
        wtime = MPI_Wtime();

        // Reduce_scatter: Reducción parcial
        MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);

        // Allgatherv: Recolectar los resultados parciales
        MPI_Allgatherv(recvbuf, recvcounts[mpi_rank], MPI_CHAR, finalbuf,
                       recvcounts, displs, MPI_CHAR, MPI_COMM_WORLD);

        wtime = MPI_Wtime() - wtime;
        wtime_sum += wtime;
    }

    // Calcular el tiempo promedio
    wtime_sum /= iterations;
    if (mpi_rank == 0) {
        printf("Processes: %d, Sendcount: %d, Latency: %e seconds\n", mpi_size, sendcount, wtime_sum);
    }

    // Liberar recursos
    free(sendbuf);
    free(recvbuf);
    free(finalbuf);
    free(recvcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}

