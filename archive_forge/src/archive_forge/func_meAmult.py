import sys
import numpy as np
from itertools import combinations_with_replacement
import ase.units as u
from ase.parallel import parprint, paropen
from ase.vibrations.resonant_raman import ResonantRaman
from ase.vibrations.franck_condon import FranckCondonOverlap
from ase.vibrations.franck_condon import FranckCondonRecursive
def meAmult(self, omega, gamma=0.1):
    """Evaluate Albrecht A term.

        Returns
        -------
        Full Albrecht A matrix element. Unit: e^2 Angstrom^2 / eV
        """
    self.read()
    if not hasattr(self, 'fcr'):
        self.fcr = FranckCondonRecursive()
    omL = omega + 1j * gamma
    omS_v = omL - self.om_v
    nv = len(self.om_v)
    om_Q = self.om_Q[self.skip:]
    nQ = len(om_Q)
    n_v = self.d_vQ.sum(axis=1)
    nvib_ov = np.empty((self.combinations, nv), dtype=int)
    om_ov = np.zeros((self.combinations, nv), dtype=float)
    n_ov = np.zeros((self.combinations, nv), dtype=int)
    d_ovQ = np.zeros((self.combinations, nv, nQ), dtype=int)
    for o in range(self.combinations):
        nvib_ov[o] = np.array(n_v == o + 1)
        for v in range(nv):
            try:
                om_ov[o, v] = om_Q[self.ind_v[v][o]]
                d_ovQ[o, v, self.ind_v[v][o]] = 1
            except IndexError:
                pass
    n_ov[0] = self.n_vQ.max(axis=1)
    n_ov[1] = nvib_ov[1]
    n_p, myp, exF_pr = self.init_parallel_excitations()
    m_vcc = np.zeros((nv, 3, 3), dtype=complex)
    for p in myp:
        energy = self.ex0E_p[p]
        d_Q = self.unitless_displacements(exF_pr[p])[self.skip:]
        S_Q = d_Q ** 2 / 2.0
        energy_v = energy - self.d_vQ.dot(om_Q * S_Q)
        me_cc = np.outer(self.ex0m_pc[p], self.ex0m_pc[p].conj())
        fco1_mQ = np.empty((self.nm, nQ), dtype=float)
        fco2_mQ = np.empty((self.nm, nQ), dtype=float)
        for m in range(self.nm):
            fco1_mQ[m] = self.fcr.direct0mm1(m, d_Q)
            fco2_mQ[m] = self.fcr.direct0mm2(m, d_Q)
        wm_v = np.zeros(nv, dtype=complex)
        wp_v = np.zeros(nv, dtype=complex)
        for m in range(self.nm):
            fco1_v = np.where(n_ov[0] == 2, d_ovQ[0].dot(fco2_mQ[m]), d_ovQ[0].dot(fco1_mQ[m]))
            em_v = energy_v + m * om_ov[0]
            fco_v = nvib_ov[0] * fco1_v
            wm_v += fco_v / (em_v - omL)
            wp_v += fco_v / (em_v + omS_v)
            if nvib_ov[1].any():
                for n in range(self.nm):
                    fco2_v = d_ovQ[1].dot(fco1_mQ[n])
                    e_v = em_v + n * om_ov[1]
                    fco_v = nvib_ov[1] * fco1_v * fco2_v
                    wm_v += fco_v / (e_v - omL)
                    wp_v += fco_v / (e_v + omS_v)
        m_vcc += np.einsum('a,bc->abc', wm_v, me_cc)
        m_vcc += np.einsum('a,bc->abc', wp_v, me_cc.conj())
    self.comm.sum(m_vcc)
    return m_vcc