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
def multipath_reconfigure(self):
    """Issue a multipathd reconfigure.

        When attachments come and go, the multipathd seems
        to get lost and not see the maps.  This causes
        resize map to fail 100%.  To overcome this we have
        to issue a reconfigure prior to resize map.
        """
    out, _err = self._execute('multipathd', 'reconfigure', run_as_root=True, root_helper=self._root_helper)
    return out