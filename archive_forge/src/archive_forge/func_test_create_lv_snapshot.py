from unittest import mock
from oslo_concurrency import processutils
from os_brick import exception
from os_brick import executor as os_brick_executor
from os_brick.local_dev import lvm as brick
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
def test_create_lv_snapshot(self):
    self.assertIsNone(self.vg.create_lv_snapshot('snapshot-1', 'fake-1'))
    with mock.patch.object(self.vg, 'get_volume', return_value=None):
        try:
            self.vg.create_lv_snapshot('snapshot-1', 'fake-non-existent')
        except exception.VolumeDeviceNotFound as e:
            self.assertEqual('fake-non-existent', e.kwargs['device'])
        else:
            self.fail('Exception not raised')