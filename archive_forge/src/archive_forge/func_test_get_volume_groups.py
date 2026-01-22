from unittest import mock
from oslo_concurrency import processutils
from os_brick import exception
from os_brick import executor as os_brick_executor
from os_brick.local_dev import lvm as brick
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
def test_get_volume_groups(self):
    self.assertEqual(3, len(self.vg.get_all_volume_groups('sudo')))
    self.assertEqual(1, len(self.vg.get_all_volume_groups('sudo', 'fake-vg')))