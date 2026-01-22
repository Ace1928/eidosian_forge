import pickle
import pytest
from netaddr import IPAddress, IPNetwork
def test_iterhosts_v6():
    assert list(IPNetwork('::ffff:192.0.2.0/125').iter_hosts()) == [IPAddress('::ffff:192.0.2.1'), IPAddress('::ffff:192.0.2.2'), IPAddress('::ffff:192.0.2.3'), IPAddress('::ffff:192.0.2.4'), IPAddress('::ffff:192.0.2.5'), IPAddress('::ffff:192.0.2.6'), IPAddress('::ffff:192.0.2.7')]