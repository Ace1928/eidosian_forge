import pickle
import pytest
from sklearn.utils.deprecation import _is_deprecated, deprecated
def test_is_deprecated():
    assert _is_deprecated(MockClass1.__new__)
    assert _is_deprecated(MockClass2().method)
    assert _is_deprecated(MockClass3.__init__)
    assert not _is_deprecated(MockClass4.__init__)
    assert _is_deprecated(MockClass5.__new__)
    assert _is_deprecated(mock_function)