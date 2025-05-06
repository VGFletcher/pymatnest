from ase import Atoms
import numpy as np
from replica_exchange_utils import *

num_replicas = 4
n_walkers = 12


def create_dummy_at(
    n_atoms,
    # do_velocities = True,
    do_GMC = True,
    n_extra_data = 0,
    # swap_atomic_numbers = True,
    track_configs = True,
    index=0
):
    at = Atoms(f"H{n_atoms}", pbc=True, positions=np.ones((n_atoms,3)), cell=np.eye(3))

    at.info["ns_energy"] = 42.3
    at.info["index"] = index
    if do_GMC:
        at.arrays["GMC_direction"] = at.positions
    if n_extra_data > 0:
        at.arrays["ns_extra_data"] = np.ones((n_atoms, n_extra_data))
    if track_configs:
        at.info['config_ind'] = index
        at.info['from_config_ind'] = 0
        at.info['config_ind_time'] = 0
    
    return at


def test_snd_and_recv_buffers():
    """Create send buffer and immadiately read it again by the corresponding 
    function, without any MPI communication. Just to check whether the 
    serialization and the deserialization works properly"""
    n_atoms = 4
    do_velocities = True
    do_GMC = True
    n_extra_data = 0
    swap_atomic_numbers = True
    track_configs = True

    at = create_dummy_at(
        n_atoms,
        # do_velocities,
        do_GMC,
        n_extra_data,
        # swap_atomic_numbers,
        track_configs,
        index=42
    )
    n_send = get_buffer_size(
        n_atoms,
        do_velocities,
        do_GMC,
        n_extra_data,
        swap_atomic_numbers,
        track_configs,
    )
    snd_buf = construct_snd_buf(
        at,
        n_send,
        n_atoms,
        do_velocities,
        do_GMC,
        n_extra_data,
        swap_atomic_numbers,
        track_configs,
    )

    at2 = create_dummy_at(
        n_atoms,
        # do_velocities,
        do_GMC,
        n_extra_data,
        # swap_atomic_numbers,
        track_configs,
        index=1
    )

    at_read = read_rcv_buf(
        at2,
        snd_buf,
        n_atoms,
        do_velocities,
        do_GMC,
        n_extra_data,
        swap_atomic_numbers,
        track_configs
    )

    assert (at == at_read)
    assert (at.info["config_ind"] == at_read.info["config_ind"])

    print(at.info)
    print(at_read.info)