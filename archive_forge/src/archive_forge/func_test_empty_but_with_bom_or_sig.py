from charset_normalizer.api import from_bytes
from charset_normalizer.models import CharsetMatches
import pytest
@pytest.mark.parametrize('payload, expected_encoding', [(b'\xfe\xff', 'utf_16'), ('\ufeff'.encode('gb18030'), 'gb18030'), (b'\xef\xbb\xbf', 'utf_8'), (''.encode('utf_32'), 'utf_32')])
def test_empty_but_with_bom_or_sig(payload, expected_encoding):
    best_guess = from_bytes(payload).best()
    assert best_guess is not None, 'Empty detection but with SIG/BOM has failed!'
    assert best_guess.encoding == expected_encoding, 'Empty detection but with SIG/BOM is wrongly detected!'
    assert best_guess.raw == payload, 'The RAW property should contain the original payload given for detection.'
    assert best_guess.byte_order_mark is True, 'The BOM/SIG property should return True'
    assert str(best_guess) == '', 'The cast to str SHOULD be empty'