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
@staticmethod
def get_all_volume_groups(root_helper: str, vg_name: Optional[str]=None) -> list[dict[str, Any]]:
    """Static method to get all VGs on a system.

        :param root_helper: root_helper to use for execute
        :param vg_name: optional, gathers info for only the specified VG
        :returns: List of Dictionaries with VG info

        """
    cmd = LVM.LVM_CMD_PREFIX + ['vgs', '--noheadings', '--unit=g', '-o', 'name,size,free,lv_count,uuid', '--separator', ':', '--nosuffix']
    if vg_name is not None:
        cmd.append(vg_name)
    out, _err = priv_rootwrap.execute(*cmd, root_helper=root_helper, run_as_root=True)
    vg_list = []
    if out is not None:
        vgs = out.split()
        for vg in vgs:
            fields = vg.split(':')
            vg_list.append({'name': fields[0], 'size': float(fields[1]), 'available': float(fields[2]), 'lv_count': int(fields[3]), 'uuid': fields[4]})
    return vg_list