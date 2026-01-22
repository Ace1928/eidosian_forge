import array
import pytest
import rpy2.rinterface_lib.sexp
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import conversion
def test_NameClassMapContext():
    ncm = conversion.NameClassMap(object)
    assert not len(ncm._map)
    with conversion.NameClassMapContext(ncm, {}):
        assert not len(ncm._map)
    assert not len(ncm._map)
    with conversion.NameClassMapContext(ncm, {'A': list}):
        assert set(ncm._map.keys()) == set('A')
    assert not len(ncm._map)
    ncm['B'] = tuple
    with conversion.NameClassMapContext(ncm, {'A': list}):
        assert set(ncm._map.keys()) == set('AB')
    assert set(ncm._map.keys()) == set('B')
    with conversion.NameClassMapContext(ncm, {'B': list}):
        assert set(ncm._map.keys()) == set('B')
    assert set(ncm._map.keys()) == set('B')
    assert ncm['B'] is tuple