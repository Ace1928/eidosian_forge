import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def test_copy_docstring_non_existing():

    def func2():
        pass
    with pytest.raises(AttributeError):
        _helpers.copy_docstring(SourceClass)(func2)