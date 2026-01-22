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
def create_raid(self, drives: list[str], raid_type: str, device_name: str, name: str, read_only: bool) -> None:
    cmd = ['mdadm']
    num_drives = len(drives)
    cmd.append('-C')
    if read_only:
        cmd.append('-o')
    cmd.append(device_name)
    cmd.append('-R')
    if name:
        cmd.append('-N')
        cmd.append(name)
    cmd.append('--level')
    cmd.append(raid_type)
    cmd.append('--raid-devices=' + str(num_drives))
    cmd.append('--bitmap=internal')
    cmd.append('--homehost=any')
    cmd.append('--failfast')
    cmd.append('--assume-clean')
    for i in range(len(drives)):
        cmd.append(drives[i])
    LOG.debug('[!] cmd = %s', cmd)
    self.run_mdadm(cmd)
    for i in range(60):
        try:
            is_exist = os.path.exists(RAID_PATH + name)
            LOG.debug('[!] md is_exist = %s', is_exist)
            if is_exist:
                return
            time.sleep(1)
        except Exception:
            LOG.debug('[!] Exception_wait_raid!')
    msg = _('md: /dev/md/%s not found.') % name
    LOG.error(msg)
    raise exception.NotFound(message=msg)