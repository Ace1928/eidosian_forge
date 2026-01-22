import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def test_update_query_remove_param():
    base_uri = 'http://www.google.com'
    uri = base_uri + '?x=a'
    updated = _helpers.update_query(uri, {'y': 'c'}, remove=['x'])
    _assert_query(updated, {'y': ['c']})