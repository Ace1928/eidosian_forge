import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def test_raw_score(atoms):
    """Test that raw_score can be extracted."""
    err_msg = "raw_score not put in atoms.info['key_value_pairs']"
    assert 'raw_score' in atoms.info['key_value_pairs'], err_msg