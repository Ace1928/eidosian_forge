import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
def test__symlink_name_from_device_path(self):
    """Get symlink for non replicated device."""
    dev_name = '/dev/nvme0n1'
    res = utils._symlink_name_from_device_path(dev_name)
    self.assertEqual('/dev/disk/by-id/os-brick+dev+nvme0n1', res)