import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.tip3p import rOH, angleHOH, TIP3P
def energy_and_forces(self, a, xpos, position_list, q_v, nmol, t, dtdd):
    """ energy and forces on molecule a from all other molecules.
            cutoff is based on O-O Distance. """
    epsil = np.tile([epsilon0], nmol - 1 - a)
    sigma = np.tile([sigma0], nmol - 1 - a)
    DOO = position_list[::4] - xpos[a * 4]
    d2 = (DOO ** 2).sum(1)
    d = np.sqrt(d2)
    e_lj = 4 * epsil * (sigma ** 12 / d ** 12 - sigma ** 6 / d ** 6)
    f_lj = (4 * epsil * (12 * sigma ** 12 / d ** 13 - 6 * sigma ** 6 / d ** 7) * t - e_lj * dtdd)[:, np.newaxis] * DOO / d[:, np.newaxis]
    self.forces[a * 4] -= f_lj.sum(0)
    self.forces[(a + 1) * 4::4] += f_lj
    e_elec = 0
    all_cut = np.repeat(t, 4)
    for i in range(4):
        D = position_list - xpos[a * 4 + i]
        d2_all = (D ** 2).sum(axis=1)
        d_all = np.sqrt(d2_all)
        e = k_c * q_v[i] * q_v / d_all
        e_elec += np.dot(all_cut, e).sum()
        e_f = e.reshape(nmol - a - 1, 4).sum(1)
        F = (e / d_all * all_cut)[:, np.newaxis] * D / d_all[:, np.newaxis]
        FOO = -(e_f * dtdd)[:, np.newaxis] * DOO / d[:, np.newaxis]
        self.forces[(a + 1) * 4 + 0::4] += FOO
        self.forces[a * 4] -= FOO.sum(0)
        self.forces[(a + 1) * 4:] += F
        self.forces[a * 4 + i] -= F.sum(0)
    self.energy += np.dot(e_lj, t) + e_elec