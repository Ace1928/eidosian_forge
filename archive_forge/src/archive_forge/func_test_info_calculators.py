import pytest
from ase.build import bulk
from ase.io import write
def test_info_calculators(cli):
    assert 'gpaw' in cli.ase('info', '--calculators')