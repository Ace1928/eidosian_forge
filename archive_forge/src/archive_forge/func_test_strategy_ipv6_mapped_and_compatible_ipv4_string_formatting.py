import platform
import pytest
from netaddr import AddrFormatError
from netaddr.strategy import ipv6
def test_strategy_ipv6_mapped_and_compatible_ipv4_string_formatting():
    assert ipv6.int_to_str(16777215) == '::0.255.255.255'
    assert ipv6.int_to_str(4294967295) == '::255.255.255.255'
    assert ipv6.int_to_str(8589934591) == '::1:ffff:ffff'
    assert ipv6.int_to_str(281474976710655) == '::ffff:255.255.255.255'
    assert ipv6.int_to_str(281470681743359) == '::fffe:ffff:ffff'
    assert ipv6.int_to_str(281474976710655) == '::ffff:255.255.255.255'
    assert ipv6.int_to_str(281474976710641) == '::ffff:255.255.255.241'
    assert ipv6.int_to_str(281474976710654) == '::ffff:255.255.255.254'
    assert ipv6.int_to_str(281474976710400) == '::ffff:255.255.255.0'
    assert ipv6.int_to_str(281474976645120) == '::ffff:255.255.0.0'
    assert ipv6.int_to_str(281474959933440) == '::ffff:255.0.0.0'
    assert ipv6.int_to_str(1099494850560) == '::ff:ff00:0'
    assert ipv6.int_to_str(562945658454016) == '::1:ffff:0:0'
    if platform.system() == 'Windows':
        assert ipv6.int_to_str(281470681743360) == '::ffff:0:0'
    else:
        assert ipv6.int_to_str(281470681743360) == '::ffff:0.0.0.0'