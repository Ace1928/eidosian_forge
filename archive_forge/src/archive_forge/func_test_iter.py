import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_iter():
    new_env = rinterface.globalenv.find('new.env')
    env = new_env()
    env['a'] = rinterface.IntSexpVector([123])
    env['b'] = rinterface.IntSexpVector([456])
    symbols = [x for x in env]
    assert len(symbols) == 2
    for s in ['a', 'b']:
        assert s in symbols