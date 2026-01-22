import os
import nibabel as nb
import numpy as np
import pytest
from ...testing import utils
from ..confounds import CompCor, TCompCor, ACompCor
@pytest.mark.parametrize('a, b, close', [([[0.1, 0.2], [0.3, 0.4]], [[-0.1, 0.2], [-0.3, 0.4]], True), ([[0.1, 0.2], [0.3, 0.4]], [[-0.1, 0.2], [0.3, -0.4]], False)])
def test_close_up_to_column_sign(a, b, close):
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    assert close_up_to_column_sign(a, b) == close
    assert close_up_to_column_sign(a, -b) == close
    assert close_up_to_column_sign(-a, b) == close
    assert close_up_to_column_sign(-a, -b) == close
    assert close_up_to_column_sign(a, a)
    assert close_up_to_column_sign(b, b)