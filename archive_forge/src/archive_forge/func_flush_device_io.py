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
def flush_device_io(self, device):
    """This is used to flush any remaining IO in the buffers."""
    if os.path.exists(device):
        try:
            LOG.debug('Flushing IO for device %s', device)
            self._execute('blockdev', '--flushbufs', device, run_as_root=True, attempts=3, timeout=300, interval=10, root_helper=self._root_helper)
        except putils.ProcessExecutionError as exc:
            LOG.warning('Failed to flush IO buffers prior to removing device: %(code)s', {'code': exc.exit_code})
            raise