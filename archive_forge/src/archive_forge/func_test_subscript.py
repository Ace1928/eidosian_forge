import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_subscript():
    ge = rinterface.globalenv
    obj = ge.find('letters')
    ge['a'] = obj
    a = ge['a']
    assert ge.find('identical')(obj, a)