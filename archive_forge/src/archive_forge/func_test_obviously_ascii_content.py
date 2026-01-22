from charset_normalizer.api import from_bytes
from charset_normalizer.models import CharsetMatches
import pytest
@pytest.mark.parametrize('payload', [b'AbAdZ pOoooOlDl mmlDoDkA lldDkeEkddA mpAlkDF', b'g4UsPJdfzNkGW2jwmKDGDilKGKYtpF2X.mx3MaTWL1tL7CNn5U7DeCcodKX7S3lwwJPKNjBT8etY', b'{"token": "g4UsPJdfzNkGW2jwmKDGDilKGKYtpF2X.mx3MaTWL1tL7CNn5U7DeCcodKX7S3lwwJPKNjBT8etY"}', b'81f4ab054b39cb0e12701e734077d84264308f5fc79494fc5f159fa2ebc07b73c8cc0e98e009664a20986706f90146e8eefcb929ce1f74a8eab21369fdc70198', b'{}'])
def test_obviously_ascii_content(payload):
    best_guess = from_bytes(payload).best()
    assert best_guess is not None, 'Dead-simple ASCII detection has failed!'
    assert best_guess.encoding == 'ascii', 'Dead-simple ASCII detection is wrongly detected!'