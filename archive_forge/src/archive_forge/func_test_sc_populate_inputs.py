import numpy as np
import numpy.testing as npt
from .. import rapidart as ra
from ...interfaces.base import Bunch
def test_sc_populate_inputs():
    sc = ra.StimulusCorrelation()
    inputs = Bunch(realignment_parameters=None, intensity_values=None, spm_mat_file=None, concatenated_design=None)
    assert set(sc.inputs.__dict__.keys()) == set(inputs.__dict__.keys())