from pytest import warns, raises
from ase import Atoms
from ase import neb
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
def test_raising_parallel_errors(self):
    with raises(RuntimeError, match='.*Cannot use shared calculators.*'):
        _ = neb.NEB(self.images_dummy, allow_shared_calculator=True, parallel=True)