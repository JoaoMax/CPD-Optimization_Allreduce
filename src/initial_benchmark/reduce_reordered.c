#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CORES_PER_NODE 12

int main(int argc, char **argv) {
    int mpi_size, mpi_rank, new_rank, color;
    int *recvcounts, *displs;
    int sendcount, recvcount, iterations, i;
    double wtime, wtime_sum;
    char *sendbuf, *recvbuf, *finalbuf;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // **Paso 1: Reorganización de Rangos**
    color = mpi_rank / CORES_PER_NODE; // Agrupa procesos en nodos lógicos
    MPI_Comm node_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, mpi_rank, &node_comm);
    MPI_Comm_rank(node_comm, &new_rank); // Nuevo rango dentro del nodo lógico

    // Configuración inicial
    sendcount = 1024 * mpi_size; // Tamaño total del mensaje
    recvcount = sendcount / mpi_size;
    iterations = 100;

    sendbuf = (char *)malloc(sendcount * sizeof(char));
    recvbuf = (char *)malloc(recvcount * sizeof(char));
    finalbuf = (char *)malloc(sendcount * sizeof(char));

    memset(sendbuf, 1, sendcount);
    memset(recvbuf, 0, recvcount);
    memset(finalbuf, 0, sendcount);

    // Configuración de Allgatherv
    recvcounts = (int *)malloc(mpi_size * sizeof(int));
    displs = (int *)malloc(mpi_size * sizeof(int));
    for (i = 0; i < mpi_size; i++) {
        recvcounts[i] = recvcount;
        displs[i] = i * recvcount;
    }

    // **Paso 2: Ejecución de Reduce_scatter + Allgatherv**
    wtime_sum = 0.0;
    for (i = 0; i < iterations; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        wtime = MPI_Wtime();

        // Reduce_scatter: Reducción parcial
        MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);

        // Allgatherv: Recolecta y distribuye resultados
        MPI_Allgatherv(recvbuf, recvcounts[mpi_rank], MPI_CHAR,
                       finalbuf, recvcounts, displs, MPI_CHAR, MPI_COMM_WORLD);

        wtime = MPI_Wtime() - wtime;
        wtime_sum += wtime;
    }

    // Promediar el tiempo y mostrar resultados
    wtime_sum /= iterations;
    if (mpi_rank == 0) {
        printf("Processes: %d, Reordered Rank: %d, Latency: %e seconds\n", mpi_size, new_rank, wtime_sum);
    }

    // Liberar recursos
    free(sendbuf);
    free(recvbuf);
    free(finalbuf);
    free(recvcounts);
    free(displs);
    MPI_Comm_free(&node_comm);
    MPI_Finalize();

    return 0;
}

