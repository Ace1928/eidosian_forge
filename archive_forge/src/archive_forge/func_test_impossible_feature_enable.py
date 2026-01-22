import sys, platform, re, pytest
from numpy.core._multiarray_umath import (
import numpy as np
import subprocess
import pathlib
import os
import re
def test_impossible_feature_enable(self):
    """
        Test that a RuntimeError is thrown if an impossible feature-enabling
        request is made. This includes enabling a feature not supported by the
        machine, or disabling a baseline optimization.
        """
    if self.UNAVAILABLE_FEAT is None:
        pytest.skip('There are no unavailable features to test with')
    bad_feature = self.UNAVAILABLE_FEAT
    self.env['NPY_ENABLE_CPU_FEATURES'] = bad_feature
    msg = f'You cannot enable CPU features \\({bad_feature}\\), since they are not supported by your machine.'
    err_type = 'RuntimeError'
    self._expect_error(msg, err_type)
    feats = f'{bad_feature}, {self.BASELINE_FEAT}'
    self.env['NPY_ENABLE_CPU_FEATURES'] = feats
    msg = f'You cannot enable CPU features \\({bad_feature}\\), since they are not supported by your machine.'
    self._expect_error(msg, err_type)