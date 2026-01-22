from __future__ import annotations
import functools
import inspect
import logging as py_logging
import os
import time
from typing import Any, Callable, Optional, Type, Union   # noqa: H301
import uuid as uuid_lib
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils import strutils
from os_brick import executor
from os_brick.i18n import _
from os_brick.privileged import nvmeof as priv_nvme
from os_brick.privileged import rootwrap as priv_rootwrap
import tenacity  # noqa
def get_dev_path(connection_properties, device_info):
    """Return the device that was returned when connecting a volume."""
    if device_info and device_info.get('path'):
        res = device_info['path']
    else:
        res = connection_properties.get('device_path') or ''
    return convert_str(res)