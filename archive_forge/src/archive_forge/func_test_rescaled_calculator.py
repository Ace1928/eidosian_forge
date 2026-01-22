import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.qmmm import ForceQMMM, RescaledCalculator
from ase.eos import EquationOfState
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances
def test_rescaled_calculator():
    """
    Test rescaled RescaledCalculator() by computing lattice constant
    and bulk modulus using fit to equation of state
    and comparing it to the desired values
    """
    from ase.calculators.eam import EAM
    from ase.units import GPa

    def pair_potential(r):
        """
        returns the pair potential as a equation 27 in pair_potential
        r - numpy array with the values of distance to compute the pair function
        """
        c = 3.25
        c0 = 47.1346499
        c1 = -33.7665655
        c2 = 6.2541999
        energy = (c0 + c1 * r + c2 * r ** 2.0) * (r - c) ** 2.0
        energy[r > c] = 0.0
        return energy

    def cohesive_potential(r):
        """
        returns the cohesive potential as a equation 28 in pair_potential
        r - numpy array with the values of distance to compute the pair function
        """
        d = 4.400224
        rho = (r - d) ** 2.0
        rho[r > d] = 0.0
        return rho

    def embedding_function(rho):
        """
        returns energy as a function of electronic density from eq 3
        """
        A = 1.896373
        energy = -A * np.sqrt(rho)
        return energy
    cutoff = 4.400224
    W_FS = EAM(elements=['W'], embedded_energy=np.array([embedding_function]), electron_density=np.array([[cohesive_potential]]), phi=np.array([[pair_potential]]), cutoff=cutoff, form='fs')

    def strain(at, e, calc):
        at = at.copy()
        at.set_cell((1.0 + e) * at.cell, scale_atoms=True)
        at.calc = calc
        v = at.get_volume()
        e = at.get_potential_energy()
        return (v, e)
    a0_qm = 3.18556
    C11_qm = 522
    C12_qm = 193
    B_qm = (C11_qm + 2.0 * C12_qm) / 3.0
    bulk_at = bulk('W', cubic=True)
    mm_calc = W_FS
    eps = np.linspace(-0.01, 0.01, 13)
    v_mm, E_mm = zip(*[strain(bulk_at, e, mm_calc) for e in eps])
    eos_mm = EquationOfState(v_mm, E_mm)
    v0_mm, E0_mm, B_mm = eos_mm.fit()
    B_mm /= GPa
    a0_mm = v0_mm ** (1.0 / 3.0)
    mm_r = RescaledCalculator(mm_calc, a0_qm, B_qm, a0_mm, B_mm)
    bulk_at = bulk('W', cubic=True, a=a0_qm)
    v_mm_r, E_mm_r = zip(*[strain(bulk_at, e, mm_r) for e in eps])
    eos_mm_r = EquationOfState(v_mm_r, E_mm_r)
    v0_mm_r, E0_mm_r, B_mm_r = eos_mm_r.fit()
    B_mm_r /= GPa
    a0_mm_r = v0_mm_r ** (1.0 / 3)
    assert abs((a0_mm_r - a0_qm) / a0_qm) < 0.001
    assert abs((B_mm_r - B_qm) / B_qm) < 0.001