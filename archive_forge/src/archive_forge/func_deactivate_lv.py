from __future__ import annotations
import math
import os
import re
from typing import Any, Callable, Optional  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from os_brick import exception
from os_brick import executor
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def deactivate_lv(self, name: str) -> None:
    lv_path = self.vg_name + '/' + self._mangle_lv_name(name)
    cmd = ['lvchange', '-a', 'n']
    cmd.append(lv_path)
    try:
        self._execute(*cmd, root_helper=self._root_helper, run_as_root=True)
    except putils.ProcessExecutionError as err:
        LOG.exception('Error deactivating LV')
        LOG.error('Cmd     :%s', err.cmd)
        LOG.error('StdOut  :%s', err.stdout)
        LOG.error('StdErr  :%s', err.stderr)
        raise
    self._wait_for_volume_deactivation(name)