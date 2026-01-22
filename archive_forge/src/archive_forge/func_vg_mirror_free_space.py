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
def vg_mirror_free_space(self, mirror_count: int) -> float:
    free_capacity = 0.0
    disks = []
    for pv in self.pv_list:
        disks.append(float(pv['available']))
    while True:
        disks = sorted([a for a in disks if a > 0.0], reverse=True)
        if len(disks) <= mirror_count:
            break
        disk = disks[-1]
        disks = disks[:-1]
        for index in list(range(mirror_count)):
            disks[index] -= disk
        free_capacity += disk
    return free_capacity