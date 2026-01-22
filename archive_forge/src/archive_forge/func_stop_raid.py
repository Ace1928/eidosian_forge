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
def stop_raid(self, md_path: str, raise_exception: bool=False) -> Optional[str]:
    cmd = ['mdadm', '--stop', md_path]
    LOG.debug('[!] cmd = %s', cmd)
    cmd_out = self.run_mdadm(cmd, raise_exception)
    return cmd_out