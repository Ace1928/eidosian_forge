from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def get_ddp_error_percent(_drive_count, _extent_count):
    """Determine the space reserved for reconstruction"""
    if _drive_count <= 36:
        if _extent_count <= 600:
            return 0.4
        elif _extent_count <= 1400:
            return 0.35
        elif _extent_count <= 6200:
            return 0.2
        elif _extent_count <= 50000:
            return 0.15
    elif _drive_count <= 64:
        if _extent_count <= 600:
            return 0.2
        elif _extent_count <= 1400:
            return 0.15
        elif _extent_count <= 6200:
            return 0.1
        elif _extent_count <= 50000:
            return 0.05
    elif _drive_count <= 480:
        if _extent_count <= 600:
            return 0.2
        elif _extent_count <= 1400:
            return 0.15
        elif _extent_count <= 6200:
            return 0.1
        elif _extent_count <= 50000:
            return 0.05
    self.module.fail_json(msg='Drive count exceeded the error percent table. Array[%s]' % self.ssid)