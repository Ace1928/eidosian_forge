import platform
import pytest
from netaddr import AddrFormatError
from netaddr.strategy import ipv6
@pytest.mark.parametrize('str_value', ('FEDC:BA98:7654:3210:FEDC:BA98:7654:3210', '1080:0:0:0:8:800:200C:417A', 'FF01:0:0:0:0:0:0:43', '0:0:0:0:0:0:0:1', '0:0:0:0:0:0:0:0', '1080::8:800:200C:417A', 'FF01::43', '::1', '::', '::192.0.2.1', '::ffff:192.0.2.1', '0:0:0:0:0:0:192.0.2.1', '0:0:0:0:0:FFFF:192.0.2.1', '0:0:0:0:0:0:13.1.68.3', '0:0:0:0:0:FFFF:129.144.52.38', '::13.1.68.3', '::FFFF:129.144.52.38', '1::', '::ffff', 'ffff::', 'ffff::ffff', '0:1:2:3:4:5:6:7', '8:9:a:b:c:d:e:f', '0:0:0:0:0:0:0:0', 'ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff'))
def test_strategy_ipv6_valid_str(str_value):
    assert ipv6.valid_str(str_value)