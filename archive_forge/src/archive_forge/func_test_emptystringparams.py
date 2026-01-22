import pytest
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
def test_emptystringparams():
    d = dict([('', 1)])
    with pytest.raises(ValueError):
        rinterface.baseenv['list'](**d)