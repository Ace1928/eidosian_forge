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
@utils.retry((putils.ProcessExecutionError, exception.BrickException), retries=3)
def multipath_del_map(self, mpath):
    """Stop monitoring a multipath given its device name (eg: dm-7).

        Method ensures that the multipath device mapper actually dissapears
        from sysfs.
        """
    map_name = self.get_dm_name(mpath)
    if map_name:
        self._execute('multipathd', 'del', 'map', map_name, run_as_root=True, timeout=5, root_helper=self._root_helper)
    if map_name and self.get_dm_name(mpath):
        raise exception.BrickException("Multipath doesn't go away")
    LOG.debug('Multipath %s no longer present', mpath)