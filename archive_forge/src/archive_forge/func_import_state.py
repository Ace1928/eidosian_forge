import warnings
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor, active_new_opmath
from pennylane.pauli import pauli_sentence
from pennylane.wires import Wires
def import_state(solver, tol=1e-15):
    """Convert an external wavefunction to a state vector.

    The sources of wavefunctions that are currently accepted are listed below.

        * The PySCF library (the restricted and unrestricted CISD/CCSD
          methods are supported). The `solver` argument is then the associated PySCF CISD/CCSD Solver object.
        * The library Dice implementing the SHCI method. The `solver` argument is then the tuple(list[str], array[float]) of Slater determinants and their coefficients.
        * The library Block2 implementing the DMRG method. The `solver` argument is then the tuple(list[int], array[float]) of Slater determinants and their coefficients.

    Args:
        solver: external wavefunction object
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Raises:
        ValueError: if external object type is not supported

    Returns:
        array: normalized state vector of length :math:`2^M`, where :math:`M` is the number of spin orbitals

    **Example**

    >>> from pyscf import gto, scf, ci
    >>> mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0,0,0.71)]], basis='sto6g')
    >>> myhf = scf.UHF(mol).run()
    >>> myci = ci.UCISD(myhf).run()
    >>> wf_cisd = qml.qchem.import_state(myci, tol=1e-1)
    >>> print(wf_cisd)
    [ 0.        +0.j  0.        +0.j  0.        +0.j  0.1066467 +0.j
      0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
      0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
     -0.99429698+0.j  0.        +0.j  0.        +0.j  0.        +0.j]
    """
    method = str(solver.__str__)
    if 'CISD' in method and 'UCISD' not in method:
        wf_dict = _rcisd_state(solver, tol=tol)
    elif 'UCISD' in method:
        wf_dict = _ucisd_state(solver, tol=tol)
    elif 'CCSD' in method and 'UCCSD' not in method:
        wf_dict = _rccsd_state(solver, tol=tol)
    elif 'UCCSD' in method:
        wf_dict = _uccsd_state(solver, tol=tol)
    elif 'tuple' in method and len(solver) == 2:
        if isinstance(solver[0][0], str):
            wf_dict = _shci_state(solver, tol=tol)
        elif isinstance(solver[0][0][0], int):
            wf_dict = _dmrg_state(solver, tol=tol)
        else:
            raise ValueError('For tuple input, the supported objects are tuple(list[str], array[float]) for SHCI calculations with Dice library and tuple(list[int], array[float]) for DMRG calculations with the Block2 library.')
    else:
        raise ValueError('The supported objects are RCISD, UCISD, RCCSD, and UCCSD for restricted and unrestricted configuration interaction and coupled cluster calculations, and tuple(list[str], array[float]) for SHCI calculations with Dice library and tuple(list[int], array[float]) for DMRG calculations with the Block2 library.')
    if 'tuple' in method:
        wf = _wfdict_to_statevector(wf_dict, len(solver[0][0]))
    else:
        wf = _wfdict_to_statevector(wf_dict, solver.mol.nao)
    return wf