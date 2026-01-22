import pytest
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
def test_scalar_convert_double():
    assert 'double' == rinterface.baseenv['typeof'](1.0)[0]