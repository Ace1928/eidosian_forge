from netaddr import (
def test_glob_to_iptuple():
    assert glob_to_iptuple('*.*.*.*') == (IPAddress('0.0.0.0'), IPAddress('255.255.255.255'))