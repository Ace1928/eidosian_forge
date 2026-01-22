import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.qmmm import combine_lj_lorenz_berthelot
from ase import units
import copy
def coulomb(self, xpos1, xpos2, xc1, xc2, spm1, spm2):
    energy = 0.0
    forces = np.zeros((len(xc1) + len(xc2), 3))
    self.xpos1 = xpos1
    self.xpos2 = xpos2
    R1 = xpos1
    R2 = xpos2
    F1 = np.zeros_like(R1)
    F2 = np.zeros_like(R2)
    C1 = xc1.reshape((-1, np.shape(xpos1)[1]))
    C2 = xc2.reshape((-1, np.shape(xpos2)[1]))
    cell = self.cell.diagonal()
    for m1, (r1, c1) in enumerate(zip(R1, C1)):
        for m2, (r2, c2) in enumerate(zip(R2, C2)):
            r00 = r2[0] - r1[0]
            shift = np.zeros(3)
            for i, periodic in enumerate(self.pbc):
                if periodic:
                    L = cell[i]
                    shift[i] = (r00[i] + L / 2.0) % L - L / 2.0 - r00[i]
            r00 += shift
            d00 = (r00 ** 2).sum() ** 0.5
            t = 1
            dtdd = 0
            if d00 > self.rc:
                continue
            elif d00 > self.rc - self.width:
                y = (d00 - self.rc + self.width) / self.width
                t -= y ** 2 * (3.0 - 2.0 * y)
                dtdd = r00 * 6 * y * (1.0 - y) / (self.width * d00)
            for a1 in range(spm1):
                for a2 in range(spm2):
                    r = r2[a2] - r1[a1] + shift
                    d2 = (r ** 2).sum()
                    d = d2 ** 0.5
                    e = k_c * c1[a1] * c2[a2] / d
                    energy += t * e
                    F1[m1, a1] -= t * (e / d2) * r
                    F2[m2, a2] += t * (e / d2) * r
                    F1[m1, 0] -= dtdd * e
                    F2[m2, 0] += dtdd * e
    F1 = F1.reshape((-1, 3))
    F2 = F2.reshape((-1, 3))
    atoms1 = self.atoms1.copy()
    atoms1.calc = copy.copy(self.calc1)
    atoms1.calc.atoms = atoms1
    F1 = atoms1.calc.redistribute_forces(F1)
    atoms2 = self.atoms2.copy()
    atoms2.calc = copy.copy(self.calc2)
    atoms2.calc.atoms = atoms2
    F2 = atoms2.calc.redistribute_forces(F2)
    forces = np.zeros((len(self.atoms), 3))
    forces[self.mask] = F1
    forces[~self.mask] = F2
    return (energy, forces)