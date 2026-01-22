from pytest import warns, raises
from ase import Atoms
from ase import neb
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
def test_no_shared_calc(self):
    images_shared_calc = [self.h_atom.copy(), self.h_atom.copy(), self.h_atom.copy()]
    shared_calc = EMT()
    for at in images_shared_calc:
        at.calc = shared_calc
    neb_not_allow = neb.NEB(images_shared_calc, allow_shared_calculator=False)
    with raises(ValueError, match='.*NEB images share the same.*'):
        neb_not_allow.get_forces()
    with raises(RuntimeError, match='.*Cannot set shared calculator.*'):
        neb_not_allow.set_calculators(EMT())
    new_calculators = [EMT() for _ in range(neb_not_allow.nimages)]
    neb_not_allow.set_calculators(new_calculators)
    for i in range(neb_not_allow.nimages):
        assert new_calculators[i] == neb_not_allow.images[i].calc
    neb_not_allow.set_calculators(new_calculators[1:-1])
    for i in range(1, neb_not_allow.nimages - 1):
        assert new_calculators[i] == neb_not_allow.images[i].calc
    with raises(RuntimeError, match='.*does not fit to len.*'):
        neb_not_allow.set_calculators(new_calculators[:-1])