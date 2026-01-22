from pytest import warns, raises
from ase import Atoms
from ase import neb
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
def test_freeze_method(self):
    at = self.h_atom.copy()
    at.calc = EMT()
    at.get_forces()
    results = dict(**at.calc.results)
    neb.NEB.freeze_results_on_image(at, **results)
    assert isinstance(at.calc, SinglePointCalculator)