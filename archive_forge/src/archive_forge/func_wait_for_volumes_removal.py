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
def wait_for_volumes_removal(self, volumes_names: list[str]) -> None:
    """Wait for device paths to be removed from the system."""
    str_names = ', '.join(volumes_names)
    LOG.debug('Checking to see if SCSI volumes %s have been removed.', str_names)
    exist = ['/dev/' + volume_name for volume_name in volumes_names]
    for i in range(61):
        exist = [path for path in exist if os.path.exists(path)]
        if not exist:
            LOG.debug('SCSI volumes %s have been removed.', str_names)
            return
        if i < 60:
            time.sleep(0.5)
            if i % 10 == 0:
                LOG.debug('%s still exist.', ', '.join(exist))
    raise exception.VolumePathNotRemoved(volume_path=exist)