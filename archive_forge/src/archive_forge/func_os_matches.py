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
def os_matches(current_os: str, connector_os: str) -> bool:
    curr_os = current_os.upper()
    conn_os = connector_os.upper()
    if conn_os == 'ALL':
        return True
    if conn_os == curr_os or conn_os in curr_os:
        return True
    return False