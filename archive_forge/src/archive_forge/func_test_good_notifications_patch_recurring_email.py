from unittest import mock
from oslotest import base
from monascaclient.osc import migration as migr
from monascaclient.v2_0 import notifications
from monascaclient.v2_0 import shell
@mock.patch('monascaclient.osc.migration.make_client')
def test_good_notifications_patch_recurring_email(self, mc):
    args = '--type EMAIL --address john.doe@hpe.com --period 60'
    data = {'type': 'EMAIL', 'address': 'john.doe@hpe.com', 'period': 60}
    self._patch_test(mc, args, data)