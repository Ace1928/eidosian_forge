from pathlib import Path
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.db import connect
def test_show_values(cli, dbfile):
    txt = cli.ase('db', dbfile, '--show-values', 'oranges,carrots')
    print(txt)
    assert 'carrots: [3..4]' in txt