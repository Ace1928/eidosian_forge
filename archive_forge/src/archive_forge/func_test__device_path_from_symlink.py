import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
def test__device_path_from_symlink(self):
    """Get device name for non replicated symlink."""
    symlink = '/dev/disk/by-id/os-brick+dev+nvme0n1'
    res = utils._device_path_from_symlink(symlink)
    self.assertEqual('/dev/nvme0n1', res)