import os
import sys
import pytest
from ase.build import molecule
from ase.calculators.calculator import CalculatorSetupError, get_calculator_class
from ase.calculators.vasp import Vasp
from ase.calculators.vasp.vasp import check_atoms, check_pbc, check_cell, check_atoms_type
def test_verify_no_run():
    """Verify that we get an error if we try and execute the calculator,
    due to the fixture.
    """
    calc = Vasp()
    with pytest.raises(AssertionError):
        calc._run()