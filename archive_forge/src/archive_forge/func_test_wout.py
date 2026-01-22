import io
from ase.io import read
from ase.io.wannier90 import read_wout_all
def test_wout():
    file = io.StringIO(wout)
    hhx = read(file, format='wout')
    assert ''.join(hhx.symbols) == 'HHX'