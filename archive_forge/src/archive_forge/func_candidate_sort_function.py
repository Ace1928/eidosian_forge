from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def candidate_sort_function(entry):
    """Orders candidates based on tray/drawer loss protection."""
    preference = 3
    if entry['drawerLossProtection']:
        preference -= 1
    if entry['trayLossProtection']:
        preference -= 2
    return preference