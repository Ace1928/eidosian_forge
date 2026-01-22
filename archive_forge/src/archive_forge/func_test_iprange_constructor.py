from ast import literal_eval
import pickle
import sys
import sys
import pytest
from netaddr import (
def test_iprange_constructor():
    iprange = IPRange('192.0.2.1', '192.0.2.254')
    assert iprange == IPRange('192.0.2.1', '192.0.2.254')
    assert '%s' % iprange == '192.0.2.1-192.0.2.254'
    assert IPRange('::ffff:192.0.2.1', '::ffff:192.0.2.254') == IPRange('::ffff:192.0.2.1', '::ffff:192.0.2.254')
    assert IPRange('192.0.2.1', '192.0.2.1') == IPRange('192.0.2.1', '192.0.2.1')
    assert IPRange('208.049.164.000', '208.050.066.255', flags=ZEROFILL) == IPRange('208.49.164.0', '208.50.66.255')
    with pytest.raises(AddrFormatError):
        IPRange('192.0.2.2', '192.0.2.1')
    with pytest.raises(AddrFormatError):
        IPRange('::', '0.0.0.1')
    with pytest.raises(AddrFormatError):
        IPRange('0.0.0.0', '::1')