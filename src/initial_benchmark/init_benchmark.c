#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/shm.h>

#define CORES_PER_NODE 12
#define NUM_BARRIERS 4

static int shmemid = -1;
static char volatile *shmem = NULL;

static int barrier_count = 0;

// Función de barrera personalizada para sincronización dentro de un nodo
static void node_barrier(int num_cores) {
  __sync_fetch_and_add(shmem + barrier_count, 1);
  while (shmem[barrier_count] != num_cores) {
    ;
  }
  shmem[(barrier_count + NUM_BARRIERS - 1) % NUM_BARRIERS] = 0;
  barrier_count = (barrier_count + 1) % NUM_BARRIERS;
}

int main(int argc, char **argv) {
  char *sendbuf, *recvbuf;
  int mpi_size, mpi_rank, node_rank, node_size;
  int dest, source, sendcount, recvcount, cores, iterations, i;
  double wtime, wtime_sum;

  MPI_Request req_send, req_recv;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // Verificación de procesos
  if (mpi_size % CORES_PER_NODE != 0) {
    if (mpi_rank == 0) {
      printf("Error: El número de procesos debe ser múltiplo de CORES_PER_NODE (%d)\n", CORES_PER_NODE);
    }
    MPI_Finalize();
    exit(1);
  }

  // Dividir procesos en nodos lógicos
  MPI_Comm node_comm;
  MPI_Comm_split(MPI_COMM_WORLD, mpi_rank / CORES_PER_NODE, mpi_rank, &node_comm);
  MPI_Comm_rank(node_comm, &node_rank);
  MPI_Comm_size(node_comm, &node_size);

  // Configuración inicial
  sendcount = 1024;
  recvcount = sendcount;
  iterations = 100;

  sendbuf = (char *)malloc(sendcount);
  recvbuf = (char *)malloc(recvcount);
  memset(sendbuf, 1, sendcount);
  memset(recvbuf, 0, recvcount);

  // Configuración de destino y fuente
  dest = (mpi_rank + 1) % mpi_size;
  source = (mpi_rank - 1 + mpi_size) % mpi_size;

  MPI_Send_init(sendbuf, sendcount, MPI_CHAR, dest, 0, MPI_COMM_WORLD, &req_send);
  MPI_Recv_init(recvbuf, recvcount, MPI_CHAR, source, 0, MPI_COMM_WORLD, &req_recv);

  // Iteraciones con cores simulados
  for (cores = 1; cores <= node_size; cores++) {
    wtime_sum = 0.0;

    for (i = 0; i < iterations; i++) {
      node_barrier(node_size); // Barrera dentro del nodo
      wtime = MPI_Wtime();

      // Iniciar comunicaciones persistentes
      MPI_Start(&req_send);
      MPI_Start(&req_recv);
      MPI_Wait(&req_send, MPI_STATUS_IGNORE);
      MPI_Wait(&req_recv, MPI_STATUS_IGNORE);

      node_barrier(node_size);
      wtime = MPI_Wtime() - wtime;
      wtime_sum += wtime;
    }

    // Promediar el tiempo de comunicación
    wtime_sum /= iterations;
    if (mpi_rank == 0) {
      printf("Processes: %d, Cores (Logical): %d, Sendcount: %d, Latency: %e seconds\n",
             mpi_size, cores, sendcount, wtime_sum);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Liberar recursos
  MPI_Request_free(&req_send);
  MPI_Request_free(&req_recv);
  free(sendbuf);
  free(recvbuf);
  MPI_Comm_free(&node_comm);

  MPI_Finalize();
  return 0;
}

