from unittest import mock
from oslo_concurrency import processutils
from os_brick import exception
from os_brick import executor as os_brick_executor
from os_brick.local_dev import lvm as brick
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
def test_lv_deactivate(self):
    with mock.patch.object(self.vg, '_execute', return_value=(0, 0)):
        is_active_mock = mock.Mock()
        is_active_mock.return_value = False
        self.vg._lv_is_active = is_active_mock
        self.vg.create_volume('test', '1G')
        self.vg.deactivate_lv('test')