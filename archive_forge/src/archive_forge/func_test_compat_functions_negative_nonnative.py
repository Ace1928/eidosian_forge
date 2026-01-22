import pytest
from cheroot._compat import extract_bytes, ntob, ntou, bton
@pytest.mark.parametrize('func', (ntob, ntou))
def test_compat_functions_negative_nonnative(func):
    """Check that compatibility functions fail loudly for incorrect input."""
    non_native_test_str = b'bar'
    with pytest.raises(TypeError):
        func(non_native_test_str, encoding='utf-8')