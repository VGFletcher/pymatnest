import numpy as np
from ase import Atoms

def swap_acceptance(
    at0: Atoms, at1: Atoms, p0: float, p1: float, elim0: float, elim1: float
) -> bool:
    """Computes the swap move acceptance criterion
    """
    # Example:
    v0 = at0.get_volume()
    v1 = at1.get_volume()
    h0 = at0.info["ns_energy"]
    h1 = at1.info["ns_energy"]
    u0 = h0 - p0 * v0
    u1 = h1 - p1 * v1

    h0_new = u1 + p0 * v1
    h1_new = u0 + p1 * v0

    return (h0_new < elim0) & (h1_new < elim1)


def get_buffer_size(
    n_atoms: int,
    do_velocities: bool,
    do_GMC: bool,
    n_extra_data: int,
    swap_atomic_numbers: bool,
    track_configs: bool
):
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
