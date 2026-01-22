import pytest
from ase.calculators.gromacs import parse_gromacs_version, get_gromacs_version
def test_parse_gromacs_version():
    assert parse_gromacs_version(sample_header) == '2020.1-Ubuntu-2020.1-1'