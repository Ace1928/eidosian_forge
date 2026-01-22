import os
import re
from shutil import copyfile
import numpy as np
def hf_state(electrons, orbitals):
    """Generate the occupation-number vector representing the Hartree-Fock state.

    The many-particle wave function in the Hartree-Fock (HF) approximation is a `Slater determinant
    <https://en.wikipedia.org/wiki/Slater_determinant>`_. In Fock space, a Slater determinant
    for :math:`N` electrons is represented by the occupation-number vector:

    .. math::

        \\vert {\\bf n} \\rangle = \\vert n_1, n_2, \\dots, n_\\mathrm{orbs} \\rangle,
        n_i = \\left\\lbrace \\begin{array}{ll} 1 & i \\leq N \\\\ 0 & i > N \\end{array} \\right.,

    where :math:`n_i` indicates the occupation of the :math:`i`-th orbital.

    Args:
        electrons (int): Number of electrons. If an active space is defined, this
            is the number of active electrons.
        orbitals (int): Number of *spin* orbitals. If an active space is defined,
            this is the number of active spin-orbitals.

    Returns:
        array: NumPy array containing the vector :math:`\\vert {\\bf n} \\rangle`

    **Example**

    >>> state = hf_state(2, 6)
    >>> print(state)
    [1 1 0 0 0 0]
    """
    if electrons <= 0:
        raise ValueError(f"The number of active electrons has to be larger than zero; got 'electrons' = {electrons}")
    if electrons > orbitals:
        raise ValueError(f"The number of active orbitals cannot be smaller than the number of active electrons; got 'orbitals'={orbitals} < 'electrons'={electrons}")
    state = np.where(np.arange(orbitals) < electrons, 1, 0)
    return np.array(state)