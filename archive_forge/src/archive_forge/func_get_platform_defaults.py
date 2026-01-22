from __future__ import absolute_import, division, print_function
import json
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import PY2, PY3
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def get_platform_defaults(self):
    """Update ref with platform specific defaults"""
    plat = self.get_platform_shortname()
    if not plat:
        return
    ref = self._ref
    ref['_platform_shortname'] = plat
    for k in ref['commands']:
        if plat in ref[k].get('_exclude', ''):
            ref['commands'].remove(k)
    plat_spec_cmds = [k for k in ref['commands'] if plat in ref[k]]
    for k in plat_spec_cmds:
        for plat_key in ref[k][plat]:
            ref[k][plat_key] = ref[k][plat][plat_key]