import sys
import numpy as np
from itertools import combinations_with_replacement
import ase.units as u
from ase.parallel import parprint, paropen
from ase.vibrations.resonant_raman import ResonantRaman
from ase.vibrations.franck_condon import FranckCondonOverlap
from ase.vibrations.franck_condon import FranckCondonRecursive
def meBC(self, omega, gamma=0.1, term='BC'):
    """Evaluate Albrecht BC term.

        Returns
        -------
        Full Albrecht BC matrix element.
        Unit: e^2 Angstrom / eV / sqrt(amu)
        """
    self.read()
    if not hasattr(self, 'fco'):
        self.fco = FranckCondonOverlap()
    omL = omega + 1j * gamma
    omS_Q = omL - self.om_Q
    n_p, myp, exF_pr = self.init_parallel_excitations()
    exdmdr_rpc = self._collect_r(self.exdmdr_rpc, [n_p, 3], self.ex0m_pc.dtype)
    dmdq_qpc = (exdmdr_rpc.T * self.im_r).T
    dmdQ_Qpc = np.dot(dmdq_qpc.T, self.modes_Qq.T).T
    me_Qcc = np.zeros((self.ndof, 3, 3), dtype=complex)
    for p in myp:
        energy = self.ex0E_p[p]
        S_Q = self.Huang_Rhys_factors(exF_pr[p])
        energy_Q = energy - self.om_Q * S_Q
        m_c = self.ex0m_pc[p]
        dmdQ_Qc = dmdQ_Qpc[:, p]
        wBLS_Q = np.zeros(self.ndof, dtype=complex)
        wBSL_Q = np.zeros(self.ndof, dtype=complex)
        wCLS_Q = np.zeros(self.ndof, dtype=complex)
        wCSL_Q = np.zeros(self.ndof, dtype=complex)
        for m in range(self.nm):
            f0mmQ1_Q = self.fco.directT0(m, S_Q) + np.sqrt(2) * self.fco.direct0mm2(m, S_Q)
            f0Qmm1_Q = self.fco.direct(1, m, S_Q)
            em_Q = energy_Q + m * self.om_Q
            wBLS_Q += f0mmQ1_Q / (em_Q - omL)
            wBSL_Q += f0Qmm1_Q / (em_Q - omL)
            wCLS_Q += f0mmQ1_Q / (em_Q + omS_Q)
            wCSL_Q += f0Qmm1_Q / (em_Q + omS_Q)
        mdmdQ_Qcc = np.einsum('a,bc->bac', m_c, dmdQ_Qc.conj())
        dmdQm_Qcc = np.einsum('ab,c->abc', dmdQ_Qc, m_c.conj())
        if 'B' in term:
            me_Qcc += np.multiply(wBLS_Q, mdmdQ_Qcc.T).T
            me_Qcc += np.multiply(wBSL_Q, dmdQm_Qcc.T).T
        if 'C' in term:
            me_Qcc += np.multiply(wCLS_Q, mdmdQ_Qcc.T).T
            me_Qcc += np.multiply(wCSL_Q, dmdQm_Qcc.T).T
    self.comm.sum(me_Qcc)
    return me_Qcc