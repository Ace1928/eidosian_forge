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
def multipath_add_path(self, realpath):
    """Add a path to multipathd for monitoring.

        This has the effect of multipathd checking an already checked device
        for multipath.

        Together with `multipath_add_wwid` we can create a multipath when
        there's only 1 path.
        """
    stdout, stderr = self._execute('multipathd', 'add', 'path', realpath, run_as_root=True, timeout=5, check_exit_code=False, root_helper=self._root_helper)
    return stdout.strip() == 'ok'