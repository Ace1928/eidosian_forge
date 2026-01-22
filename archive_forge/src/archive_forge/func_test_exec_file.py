import pytest
from ase.build import bulk, molecule
from ase.io import write
def test_exec_file(cli, images, fnameimages, execfilename):
    out = cli.ase('exec', fnameimages, '-E', execfilename)
    out_expected = [str(len(atoms)) for atoms in images]
    assert out.split() == out_expected