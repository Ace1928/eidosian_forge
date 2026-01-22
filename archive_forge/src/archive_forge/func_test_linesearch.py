import pytest
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS, BFGSLineSearch
from ase.optimize.precon import Exp, PreconLBFGS
@pytest.mark.parametrize('optcls, name', zip(optimizers, labels))
def test_linesearch(optcls, name, atoms, positions):
    maxstep = 0.2
    kwargs = {'maxstep': maxstep, 'logfile': None}
    if 'Precon' in name:
        kwargs['precon'] = Exp(A=3)
        kwargs['use_armijo'] = 'Armijo' in name
    with optcls(atoms, **kwargs) as opt:
        opt.run(steps=1)
    dr = atoms.get_positions() - positions
    steplengths = (dr ** 2).sum(1) ** 0.5
    longest_step = np.max(steplengths)
    assert longest_step < maxstep + 1e-08