import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
def test_odr_model_default_meta(self):

    def func(b, x):
        return b[0] + b[1] * x
    p = Model(func)
    p.set_meta(name='Sample Model Meta', ref='ODRPACK')
    assert_equal(p.meta, {'name': 'Sample Model Meta', 'ref': 'ODRPACK'})