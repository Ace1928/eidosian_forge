import os
import nibabel as nb
import numpy as np
import pytest
from ...testing import utils
from ..confounds import CompCor, TCompCor, ACompCor
def test_tcompcor_multi_mask_no_index(self):
    interface = TCompCor(realigned_file=self.realigned_file, mask_files=self.mask_files)
    with pytest.raises(ValueError):
        interface.run()