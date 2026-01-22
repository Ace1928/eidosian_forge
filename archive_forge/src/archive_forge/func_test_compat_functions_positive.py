import pytest
from cheroot._compat import extract_bytes, ntob, ntou, bton
@pytest.mark.parametrize(('func', 'inp', 'out'), ((ntob, 'bar', b'bar'), (ntou, 'bar', u'bar'), (bton, b'bar', 'bar')))
def test_compat_functions_positive(func, inp, out):
    """Check that compatibility functions work with correct input."""
    assert func(inp, encoding='utf-8') == out