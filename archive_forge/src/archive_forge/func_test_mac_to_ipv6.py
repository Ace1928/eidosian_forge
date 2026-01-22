import pickle
import random
from netaddr import (
def test_mac_to_ipv6():
    mac = EUI('00-1B-77-49-54-FD')
    eui = mac.eui64()
    assert mac == EUI('00-1B-77-49-54-FD')
    assert eui == EUI('00-1B-77-FF-FE-49-54-FD')
    assert mac.modified_eui64() == EUI('02-1B-77-FF-FE-49-54-FD')
    assert mac.ipv6_link_local() == IPAddress('fe80::21b:77ff:fe49:54fd')
    assert eui.ipv6_link_local() == IPAddress('fe80::21b:77ff:fe49:54fd')
    assert mac.ipv6(24196103360772296748952112894165647360) == IPAddress('1234::21b:77ff:fe49:54fd')
    assert eui.ipv6(24196103360772296748952112894165647360) == IPAddress('1234::21b:77ff:fe49:54fd')