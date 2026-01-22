import os
import numpy as np
from ...testing import utils
from .. import nilearn as iface
from ...pipeline import engine as pe
import pytest
import numpy.testing as npt
def test_signal_extr_equiv_4d_no_shared(self):
    self._test_4d_label(self.base_wanted, self.fake_equiv_4d_label_data, incl_shared_variance=False)