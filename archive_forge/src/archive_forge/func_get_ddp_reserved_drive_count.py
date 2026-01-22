from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def get_ddp_reserved_drive_count(_disk_count):
    """Determine the number of reserved drive."""
    reserve_count = 0
    if self.reserve_drive_count:
        reserve_count = self.reserve_drive_count
    elif _disk_count >= 256:
        reserve_count = 8
    elif _disk_count >= 192:
        reserve_count = 7
    elif _disk_count >= 128:
        reserve_count = 6
    elif _disk_count >= 64:
        reserve_count = 4
    elif _disk_count >= 32:
        reserve_count = 3
    elif _disk_count >= 12:
        reserve_count = 2
    elif _disk_count == 11:
        reserve_count = 1
    return reserve_count