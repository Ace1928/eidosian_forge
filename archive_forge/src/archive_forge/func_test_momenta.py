from ase import Atom, Atoms
def test_momenta(numbers=numbers, momenta=dummy_array):
    kw = {'momenta': momenta, 'velocities': momenta}
    _test_keywords(numbers=numbers, **kw)