from unittest import mock
from oslo_concurrency import processutils
from os_brick import exception
from os_brick import executor as os_brick_executor
from os_brick.local_dev import lvm as brick
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
def test_get_vg_uuid(self):
    self.assertEqual('kVxztV-dKpG-Rz7E-xtKY-jeju-QsYU-SLG6Z1', self.vg._get_vg_uuid()[0])