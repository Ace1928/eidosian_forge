import pytest
from ase.io.cif import read_cif
def test_pycodcif():
    pytest.importorskip('pycodcif')
    with open('myfile.cif', 'wb') as temp:
        temp.write(cif.encode('latin-1'))
    with open('myfile.cif') as temp:
        cif_ase = read_cif(temp, 0, reader='ase')
        cif_pycodcif = read_cif(temp, 0, reader='pycodcif')
        assert [repr(x) for x in cif_ase] == [repr(x) for x in cif_pycodcif]