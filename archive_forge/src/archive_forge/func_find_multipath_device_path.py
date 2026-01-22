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
def find_multipath_device_path(self, wwn):
    """Look for the multipath device file for a volume WWN.

        Multipath devices can show up in several places on
        a linux system.

        1) When multipath friendly names are ON:
            a device file will show up in
            /dev/disk/by-id/dm-uuid-mpath-<WWN>
            /dev/disk/by-id/dm-name-mpath<N>
            /dev/disk/by-id/scsi-mpath<N>
            /dev/mapper/mpath<N>

        2) When multipath friendly names are OFF:
            /dev/disk/by-id/dm-uuid-mpath-<WWN>
            /dev/disk/by-id/scsi-<WWN>
            /dev/mapper/<WWN>

        """
    LOG.info('Find Multipath device file for volume WWN %(wwn)s', {'wwn': wwn})
    wwn_dict = {'wwn': wwn}
    path = '/dev/disk/by-id/dm-uuid-mpath-%(wwn)s' % wwn_dict
    try:
        self.wait_for_path(path)
        return path
    except exception.VolumeDeviceNotFound:
        pass
    path = '/dev/mapper/%(wwn)s' % wwn_dict
    try:
        self.wait_for_path(path)
        return path
    except exception.VolumeDeviceNotFound:
        pass
    LOG.warning("couldn't find a valid multipath device path for %(wwn)s", wwn_dict)
    return None