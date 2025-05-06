from mpi4py import MPI
import numpy as np
from ase import Atoms
from replica_exchange_utils import *

num_replicas = 4
n_walkers = 12
n_atoms = 4
do_velocities = True
do_GMC = True
n_extra_data = 0
swap_atomic_numbers = True
track_configs = True

comm_global = MPI.COMM_WORLD
calculator_comm = MPI.COMM_SELF

rank_global = comm_global.Get_rank()  # id of process
size_global = comm_global.Get_size() 

size_replica = size_global // num_replicas
assert n_walkers % size_replica == 0
walkers_per_task = n_walkers // size_replica

replica_idx = rank_global // size_replica

# dims = [num_replicas, size_replica]
# periods = [False, False]
# comm_cart = comm_global.Create_cart(dims, periods, reorder=True)
# comm_replica = comm_cart.Sub([False, True])
comm_replica = comm_global.Split(color=replica_idx, key=0)

rank = comm_replica.Get_rank()
size = comm_replica.Get_size()
comm = comm_replica

# emulate local walkers per replica rank
all_walkers_idx = np.arange(num_replicas * n_walkers).reshape(
    num_replicas, size_replica, walkers_per_task
)
walkers = []
for i in all_walkers_idx[replica_idx, rank]:
    w = Atoms(f"H{n_atoms}", pbc=True, positions=np.ones((n_atoms,3)), cell=np.eye(3))
    w.info["index"] = i
    w.info["ns_energy"] = 42.3
    if do_GMC:
        w.arrays["GMC_direction"] = w.positions
    if n_extra_data > 0:
        w.arrays["ns_extra_data"] = np.ones((n_atoms, n_extra_data))
    if track_configs:
        w.info['config_ind'] = i
        w.info['from_config_ind'] = 0
        w.info['config_ind_time'] = 0
    walkers.append(w)

# phase 1
status = MPI.Status()
# swap_idx = np.random.randint(0, walkers_per_task)
swap_idx_phase1 = 0

n_send = get_buffer_size(
    n_atoms,
    do_velocities,
    do_GMC,
    n_extra_data,
    swap_atomic_numbers,
    track_configs
)
rcv_buf = np.zeros(n_send)
snd_buf = construct_snd_buf(
    walkers[swap_idx_phase1],
    n_send,
    n_atoms, 
    do_velocities,
    do_GMC,
    n_extra_data,
    swap_atomic_numbers,
    track_configs
)

src_rank = replica_idx * size_replica + rank
if replica_idx % 2 == 0:
    if (replica_idx + 1) < num_replicas:
        request = comm_global.Isend((snd_buf, MPI.DOUBLE), dest=src_rank + size_replica)
        comm_global.Recv((rcv_buf, MPI.DOUBLE), source=src_rank + size_replica, status=status)
        request.Wait()
        walkers[swap_idx_phase1] = read_rcv_buf(
            walkers[swap_idx_phase1], 
            rcv_buf,
            n_atoms, 
            do_velocities,
            do_GMC,
            n_extra_data,
            swap_atomic_numbers,
            track_configs
        )

if replica_idx % 2 != 0:
    request = comm_global.Irecv((rcv_buf, MPI.DOUBLE), source=src_rank - size_replica)
    comm_global.Send((snd_buf, MPI.DOUBLE), dest=src_rank - size_replica)
    request.Wait()
    walkers[swap_idx_phase1] = read_rcv_buf(
        walkers[swap_idx_phase1], 
        rcv_buf,
        n_atoms, 
        do_velocities,
        do_GMC,
        n_extra_data,
        swap_atomic_numbers,
        track_configs
    )

walkers_temp0 = np.zeros_like(all_walkers_idx)
comm_global.Gather(np.array([w.info["config_ind"] for w in walkers]), walkers_temp0, root=0)
if rank_global == 0:
    print(walkers_temp0)

# phase 2
if num_replicas > 2:
    status = MPI.Status()
    rcv_buf = np.zeros(n_send)
    # swap_idx = np.random.randint(0, walkers_per_task)
    swap_idx_phase2 = 0
    snd_buf = construct_snd_buf(
        walkers[swap_idx_phase2],
        n_send,
        n_atoms, 
        do_velocities,
        do_GMC,
        n_extra_data,
        swap_atomic_numbers,
        track_configs
    )

    src_rank = replica_idx * size_replica + rank
    if replica_idx % 2 != 0:
        if (replica_idx + 1) < num_replicas:
            comm_global.Send((snd_buf, MPI.DOUBLE), dest=src_rank + size_replica)
            comm_global.Recv((rcv_buf, MPI.DOUBLE), source=src_rank + size_replica, status=status)
            walkers[swap_idx_phase2] = read_rcv_buf(
                walkers[swap_idx_phase2], 
                rcv_buf,
                n_atoms, 
                do_velocities,
                do_GMC,
                n_extra_data,
                swap_atomic_numbers,
                track_configs
            )

    if replica_idx % 2 == 0:
        if not replica_idx == 0:
            comm_global.Recv((rcv_buf, MPI.DOUBLE), source=src_rank - size_replica, status=status)
            comm_global.Send((snd_buf, MPI.DOUBLE), dest=src_rank - size_replica)
            walkers[swap_idx_phase2] = read_rcv_buf(
                walkers[swap_idx_phase2], 
                rcv_buf,
                n_atoms, 
                do_velocities,
                do_GMC,
                n_extra_data,
                swap_atomic_numbers,
                track_configs
            )

walkers_temp0 = np.zeros_like(all_walkers_idx)
comm_global.Gather(np.array([w.info["config_ind"] for w in walkers]), walkers_temp0, root=0)
if rank_global == 0:
    print(walkers_temp0)