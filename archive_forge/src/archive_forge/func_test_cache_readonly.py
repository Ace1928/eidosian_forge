import pytest
from numpy.testing import assert_equal
from statsmodels.tools.decorators import (cache_readonly, deprecated_alias)
def test_cache_readonly():

    class Example:

        def __init__(self):
            self._cache = {}
            self.a = 0

        @cache_readonly
        def b(self):
            return 1
    ex = Example()
    assert_equal(ex.__dict__, dict(a=0, _cache={}))
    b = ex.b
    assert_equal(b, 1)
    assert_equal(ex.__dict__, dict(a=0, _cache=dict(b=1)))
    with pytest.raises(AttributeError):
        ex.b = -1
    assert_equal(ex._cache, dict(b=1))