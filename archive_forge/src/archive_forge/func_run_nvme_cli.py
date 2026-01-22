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
def run_nvme_cli(self, nvme_command: Sequence[str], **kwargs) -> tuple[str, str]:
    """Run an nvme cli command and return stdout and stderr output."""
    out, err = self._execute('nvme', *nvme_command, run_as_root=True, root_helper=self._root_helper, check_exit_code=True)
    msg = 'nvme %(nvme_command)s: stdout=%(out)s stderr=%(err)s' % {'nvme_command': nvme_command, 'out': out, 'err': err}
    LOG.debug('[!] %s', msg)
    return (out, err)