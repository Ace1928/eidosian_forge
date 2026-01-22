import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def test_string_to_scopes():
    cases = [('', []), ('a', ['a']), ('a b c d e f', ['a', 'b', 'c', 'd', 'e', 'f'])]
    for case, expected in cases:
        assert _helpers.string_to_scopes(case) == expected