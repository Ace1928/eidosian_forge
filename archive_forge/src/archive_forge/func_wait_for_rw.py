from __future__ import annotations
import glob
import os
import re
import time
from typing import Optional
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from os_brick import constants
from os_brick import exception
from os_brick import executor
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
@utils.retry(exception.BlockDeviceReadOnly, retries=5)
def wait_for_rw(self, wwn, device_path):
    """Wait for block device to be Read-Write."""
    LOG.debug('Checking to see if %s is read-only.', device_path)
    out, info = self._execute('lsblk', '-o', 'NAME,RO', '-l', '-n')
    LOG.debug('lsblk output: %s', out)
    blkdevs = out.splitlines()
    for blkdev in blkdevs:
        blkdev_parts = blkdev.split(' ')
        ro = blkdev_parts[-1]
        name = blkdev_parts[0]
        if wwn in name and int(ro) == 1:
            LOG.debug('Block device %s is read-only', device_path)
            self._execute('multipath', '-r', check_exit_code=[0, 1, 21], run_as_root=True, root_helper=self._root_helper)
            raise exception.BlockDeviceReadOnly(device=device_path)
    else:
        LOG.debug('Block device %s is not read-only.', device_path)