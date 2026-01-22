import pytest
from netaddr import INET_ATON, INET_PTON, AddrFormatError
from netaddr.strategy import ipv4
def test_strategy_inet_pton_behaviour():
    with pytest.raises(AddrFormatError):
        ipv4.str_to_int('127', flags=INET_PTON)
    with pytest.raises(AddrFormatError):
        ipv4.str_to_int('0x7f', flags=INET_PTON)
    with pytest.raises(AddrFormatError):
        ipv4.str_to_int('0177', flags=INET_PTON)
    with pytest.raises(AddrFormatError):
        ipv4.str_to_int('127.1', flags=INET_PTON)
    with pytest.raises(AddrFormatError):
        ipv4.str_to_int('0x7f.1', flags=INET_PTON)
    with pytest.raises(AddrFormatError):
        ipv4.str_to_int('0177.1', flags=INET_PTON)
    assert ipv4.str_to_int('127.0.0.1', flags=INET_PTON) == 2130706433