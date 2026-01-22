import pytest
from ase.build import bulk
from ase.io import write
def test_info_file_ok(cli, fname):
    assert 'trajectory' in cli.ase('info', fname)