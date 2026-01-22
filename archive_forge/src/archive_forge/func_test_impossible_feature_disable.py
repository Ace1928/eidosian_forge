import sys, platform, re, pytest
from numpy.core._multiarray_umath import (
import numpy as np
import subprocess
import pathlib
import os
import re
@pytest.mark.skipif(not __cpu_dispatch__, reason='NPY_*_CPU_FEATURES only parsed if `__cpu_dispatch__` is non-empty')
def test_impossible_feature_disable(self):
    """
        Test that a RuntimeError is thrown if an impossible feature-disabling
        request is made. This includes disabling a baseline feature.
        """
    if self.BASELINE_FEAT is None:
        pytest.skip('There are no unavailable features to test with')
    bad_feature = self.BASELINE_FEAT
    self.env['NPY_DISABLE_CPU_FEATURES'] = bad_feature
    msg = f"You cannot disable CPU feature '{bad_feature}', since it is part of the baseline optimizations"
    err_type = 'RuntimeError'
    self._expect_error(msg, err_type)