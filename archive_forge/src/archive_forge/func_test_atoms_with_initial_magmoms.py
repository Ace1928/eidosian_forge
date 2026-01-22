import pytest
from unittest import mock
import numpy as np
from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.calculators.vasp.create_input import _args_without_comment
from ase.calculators.vasp.create_input import _to_vasp_bool, _from_vasp_bool
from ase.build import bulk
def test_atoms_with_initial_magmoms(magmoms_factory, nacl, vaspinput_factory, assert_magmom_equal_to_incar_value, testdir):
    """Test passing atoms with initial magnetic moments"""
    magmom = magmoms_factory(nacl)
    assert len(magmom) == len(nacl)
    nacl.set_initial_magnetic_moments(magmom)
    vaspinput = vaspinput_factory(atoms=nacl)
    assert vaspinput.spinpol
    assert_magmom_equal_to_incar_value(nacl, magmom, vaspinput)