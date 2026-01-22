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
def get_device_path_by_initial_devices(self) -> Optional[str]:
    """Find target's device path from devices that were present before."""
    ctrls = [p.controller for p in self.portals if p.controller]

    def discard(devices):
        """Discard devices that don't belong to our controllers."""
        if not devices:
            return set()
        return set((dev for dev in devices if os.path.basename(dev).rsplit('n', 1)[0] in ctrls))
    current_devices = self._get_nvme_devices()
    LOG.debug('Initial devices: %s. Current devices %s. Controllers: %s', self.devices_on_start, current_devices, ctrls)
    devices = discard(current_devices) - discard(self.devices_on_start)
    if not devices:
        return None
    if len(devices) == 1:
        return devices.pop()
    if len(devices) > 1 and 1 < len(set((blk_property('uuid', os.path.basename(d)) for d in devices))):
        msg = _('Too many different volumes found for %s') % ctrls
        LOG.error(msg)
        return None
    return devices.pop()