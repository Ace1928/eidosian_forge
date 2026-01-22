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
def stop_and_assemble_raid(self, drives: list[str], md_path: str, read_only: bool) -> None:
    md_name = None
    i = 0
    assembled = False
    link = ''
    while i < 5 and (not assembled):
        for drive in drives:
            device_name = drive[5:]
            md_name = self.get_md_name(device_name)
            link = NVMeOFConnector.ks_readlink(md_path)
            if link != '':
                link = os.path.basename(link)
            if md_name and md_name == link:
                return
            LOG.debug('sleeping 1 sec -allow auto assemble link = %(link)s md path = %(md_path)s', {'link': link, 'md_path': md_path})
            time.sleep(1)
        if md_name and md_name != link:
            self.stop_raid(md_name)
        try:
            assembled = self.assemble_raid(drives, md_path, read_only)
        except Exception:
            i += 1