import os
import nibabel as nb
import numpy as np
import pytest
from ...testing import utils
from ..confounds import CompCor, TCompCor, ACompCor
@staticmethod
def fake_noise_fun(i, j, l, m):
    return m * i + l - j