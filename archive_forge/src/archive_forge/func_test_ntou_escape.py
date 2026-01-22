import pytest
from cheroot._compat import extract_bytes, ntob, ntou, bton
def test_ntou_escape():
    """Check that ``ntou`` supports escape-encoding under Python 2."""
    expected = u'hišřії'
    actual = ntou('hišřії', encoding='escape')
    assert actual == expected