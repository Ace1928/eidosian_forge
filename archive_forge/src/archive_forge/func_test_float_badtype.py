import pytest
import numpy as np
from ase.outputs import Properties, all_outputs
@pytest.mark.parametrize('val', [np.zeros(3), range(3), np.zeros(1), 1j, None])
def test_float_badtype(props, val):
    with pytest.raises(TypeError):
        props._setvalue('energy', val)