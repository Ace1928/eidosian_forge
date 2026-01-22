import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.qmmm import ForceQMMM, RescaledCalculator
from ase.eos import EquationOfState
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances
@pytest.mark.parametrize('kwargs', [{'test_size': 4, 'expected_pbc': np.array([True, True, True]), 'buffer_width': 5 * 3.61}, {'test_size': 4, 'expected_pbc': np.array([False, False, False]), 'buffer_width': 3.61}, {'test_size': [4, 4, 1], 'expected_pbc': np.array([False, False, True]), 'buffer_width': 3.61}, {'test_size': [4, 1, 4], 'expected_pbc': np.array([False, True, False]), 'buffer_width': 3.61}, {'test_size': [1, 4, 4], 'expected_pbc': np.array([True, False, False]), 'buffer_width': 3.61}, {'test_size': [1, 1, 4], 'expected_pbc': np.array([True, True, False]), 'buffer_width': 3.61}, {'test_size': [4, 1, 1], 'expected_pbc': np.array([False, True, True]), 'buffer_width': 3.61}, {'test_size': [1, 4, 1], 'expected_pbc': np.array([True, False, True]), 'buffer_width': 3.61}])
def test_qm_pbc(kwargs, qm_calc, mm_calc, bulk_at):
    kwargs1 = {}
    kwargs1.update(kwargs)
    compare_qm_cell_and_pbc(qm_calc, mm_calc, bulk_at, **kwargs1)