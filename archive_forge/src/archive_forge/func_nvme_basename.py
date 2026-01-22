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
def nvme_basename(path: str) -> str:
    """Convert a sysfs control path into a namespace device.

    We can have a basic namespace devices such as nvme0n10 which is already in
    the desired form, but there's also channels when ANA is enabled on the
    kernel which have the form nvme0c2n10 which need to be converted to
    nvme0n10 to get the actual device.
    """
    basename = os.path.basename(path)
    if 'c' not in basename:
        return basename
    ctrl, rest = basename.split('c', 1)
    ns = rest[rest.index('n'):]
    return ctrl + ns