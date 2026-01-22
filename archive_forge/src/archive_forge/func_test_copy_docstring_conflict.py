import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def test_copy_docstring_conflict():

    def func():
        """existing docstring"""
        pass
    with pytest.raises(ValueError):
        _helpers.copy_docstring(SourceClass)(func)