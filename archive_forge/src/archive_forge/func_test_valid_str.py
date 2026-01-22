import pytest
from netaddr import INET_ATON, INET_PTON, AddrFormatError
from netaddr.strategy import ipv4
@pytest.mark.parametrize(('address', 'flags', 'valid'), [['', 0, False], ['192', 0, False], ['192', INET_ATON, True], ['127.0.0.1', 0, True]])
def test_valid_str(address, flags, valid):
    assert ipv4.valid_str(address, flags) is valid