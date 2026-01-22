import os
import sys
import pytest
from ase.build import molecule
from ase.calculators.calculator import CalculatorSetupError, get_calculator_class
from ase.calculators.vasp import Vasp
from ase.calculators.vasp.vasp import check_atoms, check_pbc, check_cell, check_atoms_type
@pytest.mark.parametrize('envvar', Vasp.env_commands)
def test_make_command_envvar(envvar, monkeypatch, clear_vasp_envvar):
    """Test making a command based on the environment variables"""
    assert envvar not in os.environ
    cmd_str = 'my command'
    monkeypatch.setenv(envvar, cmd_str)
    calc = Vasp()
    cmd = calc.make_command()
    if envvar == 'VASP_SCRIPT':
        exe = sys.executable
        assert cmd == f'{exe} {cmd_str}'
    else:
        assert cmd == cmd_str