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
def flush_multipath_device(self, device_map_name):
    LOG.debug('Flush multipath device %s', device_map_name)
    self._execute('multipath', '-f', device_map_name, run_as_root=True, attempts=3, timeout=300, interval=10, root_helper=self._root_helper)