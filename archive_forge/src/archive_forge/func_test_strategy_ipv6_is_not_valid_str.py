import platform
import pytest
from netaddr import AddrFormatError
from netaddr.strategy import ipv6
@pytest.mark.parametrize('str_value', ('', 'g:h:i:j:k:l:m:n', '0:0:0:0:0:0:0:0:0'))
def test_strategy_ipv6_is_not_valid_str(str_value):
    assert not ipv6.valid_str(str_value)