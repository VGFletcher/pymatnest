from mpi4py import MPI
import numpy as np

num_replicas = 4
n_walkers = 12

comm_global = MPI.COMM_WORLD
calculator_comm = MPI.COMM_SELF

rank_global = comm_global.Get_rank()  # id of process
size_global = comm_global.Get_size() 

assert size_global % num_replicas == 0

size_replica = size_global // num_replicas
assert n_walkers % size_replica == 0
walkers_per_task = n_walkers // size_replica

replica_idx = rank_global // size_replica

dims = [num_replicas, size_replica]
periods = [False, False]
# comm_cart = comm_global.Create_cart(dims, periods, reorder=True)
# comm_replica = comm_cart.Sub([False, True])
comm_replica = comm_global.Split(color=replica_idx, key=0)

rank = comm_replica.Get_rank()
size = comm_replica.Get_size()
comm = comm_replica
# print(f"{replica_idx=}, {rank=}, {comm_replica=}")

# emulate local walkers per replica rank
all_walkers = np.arange(num_replicas * n_walkers).reshape(
    num_replicas, size_replica, walkers_per_task
)

walkers = all_walkers[replica_idx, rank]

# print(f"{(replica_idx, rank)=}", walkers)
if rank_global == 0:
    print(all_walkers)

# phase 1
status = MPI.Status()
# swap_idx = np.random.randint(0, walkers_per_task)
swap_idx = 0
rcv_buf = np.array(0, dtype=np.intc)
snd_buf = np.array(walkers[swap_idx], dtype=np.intc)

src_rank = replica_idx * size_replica + rank
if replica_idx % 2 == 0:
    if (replica_idx + 1) < num_replicas:
        # comm_global.send(snd_buf, dest=src_rank + size_replica)
        # comm_global.recv(rcv_buf, source=src_rank + size_replica, status=status)
        request = comm_global.Isend((snd_buf, 1, MPI.INT), dest=src_rank + size_replica)
        comm_global.Recv((rcv_buf, 1, MPI.INT), source=src_rank + size_replica, status=status)
        request.Wait()
        walkers[swap_idx] = rcv_buf

if replica_idx % 2 != 0:
    # comm_global.recv(rcv_buf, source=src_rank - size_replica, status=status)
    # comm_global.send(snd_buf, dest=src_rank - size_replica)
    request = comm_global.Irecv((rcv_buf, 1, MPI.INT), source=src_rank - size_replica)
    comm_global.Send((snd_buf, 1, MPI.INT), dest=src_rank - size_replica)
    request.Wait(status)
    walkers[swap_idx] = rcv_buf

walkers_temp0 = np.zeros_like(all_walkers)
comm_global.Gather(walkers, walkers_temp0, root=0)
if rank_global == 0:
    print(walkers_temp0)

# print(f"{(replica_idx, rank)=}", rcv_buf, snd_buf)

# phase 2
if num_replicas > 2:
    status = MPI.Status()
    rcv_buf = np.array(0, dtype=np.intc)
    snd_buf = np.array(walkers[0], dtype=np.intc)

    src_rank = replica_idx * size_replica + rank
    if replica_idx % 2 != 0:
        if (replica_idx + 1) < num_replicas:
            comm_global.Send((snd_buf, 1, MPI.INT), dest=src_rank + size_replica)
            comm_global.Recv((rcv_buf, 1, MPI.INT), source=src_rank + size_replica, status=status)
            walkers[swap_idx] = rcv_buf

    if replica_idx % 2 == 0:
        if not replica_idx == 0:
            comm_global.Recv((rcv_buf, 1, MPI.INT), source=src_rank - size_replica, status=status)
            comm_global.Send((snd_buf, 1, MPI.INT), dest=src_rank - size_replica)
            walkers[swap_idx] = rcv_buf

    # print(f"{(replica_idx, rank)=}", rcv_buf, snd_buf)

walkers_temp0 = np.zeros_like(all_walkers)
comm_global.Gather(walkers, walkers_temp0, root=0)
if rank_global == 0:
    print(walkers_temp0)