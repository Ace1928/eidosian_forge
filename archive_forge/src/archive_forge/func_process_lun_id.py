from __future__ import annotations
import glob
import os
import re
import time
from typing import Optional
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from os_brick import constants
from os_brick import exception
from os_brick import executor
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def process_lun_id(self, lun_ids):
    if isinstance(lun_ids, list):
        processed = []
        for x in lun_ids:
            x = self._format_lun_id(x)
            processed.append(x)
    else:
        processed = self._format_lun_id(lun_ids)
    return processed