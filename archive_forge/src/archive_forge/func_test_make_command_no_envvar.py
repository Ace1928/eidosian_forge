import os
import sys
import pytest
from ase.build import molecule
from ase.calculators.calculator import CalculatorSetupError, get_calculator_class
from ase.calculators.vasp import Vasp
from ase.calculators.vasp.vasp import check_atoms, check_pbc, check_cell, check_atoms_type
def test_make_command_no_envvar(monkeypatch, clear_vasp_envvar):
    """Test we raise when making a command with not enough information"""
    calc = Vasp()
    with pytest.raises(CalculatorSetupError):
        calc.make_command()