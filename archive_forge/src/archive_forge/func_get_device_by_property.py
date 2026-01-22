from __future__ import annotations
import errno
import functools
import glob
import json
import os.path
import time
from typing import (Callable, Optional, Sequence, Type, Union)  # noqa: H301
import uuid as uuid_lib
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def get_device_by_property(self, prop_name: str, value: str) -> Optional[str]:
    """Look for a specific device (namespace) within a controller.

        Use a specific property to identify the namespace within the
        controller and returns the device path under /dev.

        Returns None if device is not found.
        """
    LOG.debug('Looking for device where %s=%s on controller %s', prop_name, value, self.controller)
    for path in self.get_all_namespaces_ctrl_paths():
        prop_value = sysfs_property(prop_name, path)
        if prop_value == value:
            result = DEV_SEARCH_PATH + nvme_basename(path)
            LOG.debug('Device found at %s, using %s', path, result)
            return result
        LOG.debug('Block %s is not the one we are looking for (%s != %s)', path, prop_value, value)
    LOG.debug('No device Found on controller %s', self.controller)
    return None