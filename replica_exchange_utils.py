import numpy as np
from ase import Atoms

def swap_acceptance(
    at_l: Atoms, at_r: Atoms, p_l: float, p_r: float, elim_l: float, elim_r: float
) -> bool:
    """Computes the swap move acceptance criterion for configurations from 
    two replicas. We use a notation of left and right, referring to the 
    position of the replica_idx. E.g. for the swap (1 <-> 2), 1 is left, and 
    2 is right.
    
    Hence:

        Left:       Right:
        at_l         at_r
        p_l          p_r
        elim_l       elim_r
    
    We will invoke this acceptance criterion on ranks which contribute either 
    the left or the right swap partner. Thus, care needs to be taken what the
    proper inputs are for swap_acceptance.

    Consider the swap (1 <-> 2). The left partner 1 will contribute a 
    walker_snd from its pool and receive a walker_rcv from the right partner.
    Hence on the left side we need to evaluate
        
        swap_acceptance(walker_snd, walker_rcv, p1, p2, elim1, elim2)

    On the other hand, for partner 2 we have to call
    
        swap_acceptance(walker_rcv, walker_snd, p1, p2, elim1, elim2)
    """
    # Example:
    v_l = at_l.get_volume()
    v_r = at_r.get_volume()
    h_l = at_l.info["ns_energy"]
    h_r = at_r.info["ns_energy"]
    u_l = h_l - p_l * v_l
    u_r = h_r - p_r * v_r

    h_l_new = u_r + p_l * v_r
    h_r_new = u_l + p_r * v_l

    return (h_l_new < elim_l) & (h_r_new < elim_r), h_l_new, h_r_new


def get_buffer_size(
    n_atoms: int,
    do_velocities: bool,
    do_GMC: bool,
    n_extra_data: int,
    swap_atomic_numbers: bool,
    track_configs: bool
):
    """Compute the buffer size based on keywords, which determine the types
    of arrays/properties carried by a walker configuration."""
    n_send = 3*(n_atoms + 3)
    n_send += 1 # for ns_energy
    if do_velocities:
        n_send += 3*n_atoms
    if do_GMC:
        n_send += 3*n_atoms
    if n_extra_data > 0:
        n_send += n_extra_data*n_atoms
    if swap_atomic_numbers:
        n_send += n_atoms  # Z
        if do_velocities:
            n_send += n_atoms  # mass
    if track_configs:
        n_send += 3
    
    return n_send

def construct_snd_buf(
    at: Atoms,
    n_send: int,
    n_atoms: int,
    do_velocities: bool,
    do_GMC: bool,
    n_extra_data: int,
    swap_atomic_numbers: bool,
    track_configs: bool
) -> np.ndarray:
    """This is a copy of the send buffer creation used in `do_ns_loop`. It is
    only modified to collect the `ns_energy` entry of `at.info` as well.
    """
    buf = np.zeros(n_send)
    buf_o = 0
    buf[buf_o:buf_o+3*n_atoms] = at.get_positions().reshape((3*n_atoms)); buf_o += 3*n_atoms
    buf[buf_o:buf_o+3*3] = at.get_cell().reshape((3*3)); buf_o += 3*3
    if do_velocities:
        buf[buf_o:buf_o+3*n_atoms] = at.get_velocities().reshape((3*n_atoms)); buf_o += 3*n_atoms
    if do_GMC:
        buf[buf_o:buf_o+3*n_atoms] = at.arrays['GMC_direction'].reshape((3*n_atoms)); buf_o += 3*n_atoms
    if n_extra_data > 0:
        buf[buf_o:buf_o+n_extra_data*n_atoms] = at.arrays['ns_extra_data'].reshape((n_extra_data*n_atoms)); buf_o += n_extra_data*n_atoms
    if swap_atomic_numbers:
        buf[buf_o:buf_o+n_atoms] = at.get_atomic_numbers(); buf_o += n_atoms
        if do_velocities:
            buf[buf_o:buf_o+n_atoms] = at.get_masses(); buf_o += n_atoms
    if track_configs:
        buf[buf_o] = at.info['config_ind']; buf_o += 1
        buf[buf_o] = at.info['from_config_ind']; buf_o += 1
        buf[buf_o] = at.info['config_ind_time']; buf_o += 1
    buf[buf_o] = at.info['ns_energy']
    return buf
    

def read_rcv_buf(
    at: Atoms,
    buf: np.ndarray,
    n_atoms: int,
    do_velocities: bool,
    do_GMC: bool,
    n_extra_data: int,
    swap_atomic_numbers: bool,
    track_configs: bool
) -> Atoms:
    """This is a copy of the code to read the received buffer used in 
    `do_ns_loop`. It is only modified to also collect `ns_energy` into 
    `at.info`.
    """
    buf_o = 0
    at.set_positions(buf[buf_o:buf_o+3*n_atoms].reshape((n_atoms, 3))); buf_o += 3*n_atoms
    at.set_cell(buf[buf_o:buf_o+3*3].reshape((3, 3))); buf_o += 3*3
    if do_velocities:
        at.set_velocities(buf[buf_o:buf_o+3*n_atoms].reshape((n_atoms, 3))); buf_o += 3*n_atoms
    if do_GMC:
        at.arrays['GMC_direction'][:, :] = buf[buf_o:buf_o+3*n_atoms].reshape((n_atoms, 3)); buf_o += 3*n_atoms
    if n_extra_data > 0:
        at.arrays['ns_extra_data'][...] = buf[buf_o:buf_o+3*n_atoms].reshape(at.arrays['ns_extra_data'].shape); buf_o += n_extra_data*n_atoms
    if swap_atomic_numbers:
        at.set_atomic_numbers(buf[buf_o:buf_o+n_atoms].astype(int)); buf_o += n_atoms
        if do_velocities:
            at.set_masses(buf[buf_o:buf_o+n_atoms]); buf_o += n_atoms
    if track_configs:
        at.info['config_ind'] = int(buf[buf_o]); buf_o += 1
        at.info['from_config_ind'] = int(buf[buf_o]); buf_o += 1
        at.info['config_ind_time'] = int(buf[buf_o]); buf_o += 1
    at.info['ns_energy'] = float(buf[buf_o])
    return at    
