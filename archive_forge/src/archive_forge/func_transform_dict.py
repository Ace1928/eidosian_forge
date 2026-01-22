from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def transform_dict(self, data, keymap):
    transform = dict()
    for key, fact in keymap:
        if key in data:
            transform[fact] = data[key]
    return transform