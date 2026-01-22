import numpy as np
import pytest
from ase.data import s22
from ase.optimize import FIRE
from ase.constraints import FixBondLengths
from ase.calculators.tip3p import TIP3P, epsilon0, sigma0
from ase.calculators.combine_mm import CombineMM
@pytest.mark.slow
def test_combine_mm2(testdir):
    fast_test = True
    atoms = make_4mer()
    atoms.constraints = FixBondLengths([(3 * i + j, 3 * i + (j + 1) % 3) for i in range(int(len(atoms) // 3)) for j in [0, 1, 2]])
    atoms.calc = TIP3P(np.Inf)
    tag = '4mer_tip3_opt.'
    with FIRE(atoms, logfile=tag + 'log', trajectory=tag + 'traj') as opt:
        opt.run(fmax=0.05)
    tip3_pos = atoms.get_positions()
    sig = np.array([sigma0, 0, 0])
    eps = np.array([epsilon0, 0, 0])
    rc = np.Inf
    idxes = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], list(range(6)), list(range(9)), list(range(6, 12))]
    for ii, idx in enumerate(idxes):
        atoms = make_4mer()
        if fast_test:
            atoms.set_positions(tip3_pos)
        atoms.constraints = FixBondLengths([(3 * i + j, 3 * i + (j + 1) % 3) for i in range(len(atoms) // 3) for j in [0, 1, 2]])
        atoms.calc = CombineMM(idx, 3, 3, TIP3P(rc), TIP3P(rc), sig, eps, sig, eps, rc=rc)
        tag = '4mer_combtip3_opt_{0:02d}.'.format(ii)
        with FIRE(atoms, logfile=tag + 'log', trajectory=tag + 'traj') as opt:
            opt.run(fmax=0.05)
        assert (abs(atoms.positions - tip3_pos) < 1e-08).all()
        print('{0}: {1!s:>28s}: Same Geometry as TIP3P'.format(atoms.calc.name, idx))