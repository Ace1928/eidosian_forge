from unittest import mock
from oslotest import base
from oslo_privsep import capabilities
@mock.patch('oslo_privsep.capabilities._capget')
def test_get_caps(self, mock_capget):

    def impl(hdr, data):
        data[0].effective = 16908288
        data[1].effective = 131072
        data[0].permitted = 1280
        data[1].permitted = 16777224
        data[0].inheritable = 2164260864
        data[1].inheritable = 256
        return 0
    mock_capget.side_effect = impl
    self.assertCountEqual(([17, 24, 49], [8, 10, 35, 56], [24, 31, 40]), capabilities.get_caps())