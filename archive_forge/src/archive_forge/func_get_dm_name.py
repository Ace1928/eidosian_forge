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
def get_dm_name(self, dm):
    """Get the Device map name given the device name of the dm on sysfs.

        :param dm: Device map name as seen in sysfs. ie: 'dm-0'
        :returns: String with the name, or empty string if not available.
                  ie: '36e843b658476b7ed5bc1d4d10d9b1fde'
        """
    try:
        with open('/sys/block/' + dm + '/dm/name') as f:
            return f.read().strip()
    except IOError:
        return ''