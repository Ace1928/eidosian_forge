import sys
from unittest import mock
import fixtures
from oslo_concurrency import processutils
from oslo_config import cfg
from oslotest import base
from glance_store import exceptions
def test_unmount_without_mount(self):
    self._sentinel_umount()
    processutils.execute.assert_not_called()