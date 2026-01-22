import numpy as np
from ase.units import Hartree, Bohr
def polarizability(exlist, omega, form='v', tensor=False, index=0):
    """Evaluate the photon energy dependent polarizability
    from the sum over states

    Parameters
    ----------
    exlist: ExcitationList
    omega:
        Photon energy (eV)
    form: {'v', 'r'}
        Form of the dipole matrix element, default 'v'
    index: {0, 1, 2, 3}
        0: averaged, 1,2,3:alpha_xx, alpha_yy, alpha_zz, default 0
    tensor: boolean
        if True returns alpha_ij, i,j=x,y,z
        index is ignored, default False

    Returns
    -------
    alpha:
        Unit (e^2 Angstrom^2 / eV).
        Multiply with Bohr * Ha to get (Angstrom^3)
        shape = (omega.shape,) if tensor == False
        shape = (omega.shape, 3, 3) else
    """
    omega = np.asarray(omega)
    om2 = 1.0 * omega ** 2
    esc = exlist.energy_to_eV_scale
    if tensor:
        if not np.isscalar(om2):
            om2 = om2[:, None, None]
        alpha = np.zeros(omega.shape + (3, 3), dtype=om2.dtype)
        for ex in exlist:
            alpha += ex.get_dipole_tensor(form=form) / ((ex.energy * esc) ** 2 - om2)
    else:
        alpha = np.zeros_like(om2)
        for ex in exlist:
            alpha += ex.get_oscillator_strength(form=form)[index] / ((ex.energy * esc) ** 2 - om2)
    return alpha * Bohr ** 2 * Hartree