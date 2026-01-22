import numpy as np
import numpy.testing as npt
from .. import rapidart as ra
from ...interfaces.base import Bunch
def test_sc_init():
    sc = ra.StimulusCorrelation(concatenated_design=True)
    assert sc.inputs.concatenated_design