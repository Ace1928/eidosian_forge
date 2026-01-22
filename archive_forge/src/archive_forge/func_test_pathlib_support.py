from pathlib import Path
import io
import pytest
from ase.build import molecule
from ase.io import read, write
from ase.utils import PurePath, convert_string_to_fd, reader, writer
def test_pathlib_support(testdir):
    path = Path('tmp_plib_testdir')
    assert isinstance(path, PurePath)
    path.mkdir(exist_ok=True)
    myf = path / 'test.txt'
    with pytest.warns(FutureWarning):
        fd = convert_string_to_fd(myf)
        fd.close()
        assert isinstance(fd, io.TextIOBase)
    with pytest.warns(FutureWarning):
        fd = convert_string_to_fd(str(myf))
        fd.close()
        assert isinstance(fd, io.TextIOBase)
    for f in [myf, str(myf)]:
        myf.unlink()
        mywrite(f)
        myread(f)
    with myf.open('w') as fd:
        mywrite(fd, fdcmp=fd)
    with myf.open('r') as fd:
        myread(fd, fdcmp=fd)
    atoms = molecule('H2', vacuum=5)
    f2 = path / 'test2.txt'
    for form in ['vasp', 'traj', 'xyz']:
        write(f2, atoms, format=form)
        read(f2, format=form)