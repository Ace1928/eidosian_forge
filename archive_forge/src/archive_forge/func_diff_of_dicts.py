from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def diff_of_dicts(self, w, obj):
    diff = set(w.items()) - set(obj.items())
    diff = dict(diff)
    if diff and w['name'] == obj['name']:
        diff.update({'name': w['name']})
    return diff