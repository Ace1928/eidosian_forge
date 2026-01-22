import inspect
import pytest
from ..utils import deprecation
from .utils import call_method
def test_deprecation_nested1():

    def level1():
        deprecation('test message', [])
    with pytest.warns(DeprecationWarning) as record:
        call_method(level1)
    assert len(record) == 1
    assert record[0].filename == __file__