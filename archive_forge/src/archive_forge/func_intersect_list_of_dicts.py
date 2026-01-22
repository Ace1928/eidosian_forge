from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def intersect_list_of_dicts(self, w, h):
    intersect = []
    wmem = []
    hmem = []
    for d in w:
        wmem.append({'member': d['member']})
    for d in h:
        hmem.append({'member': d['member']})
    set_w = set((tuple(sorted(d.items())) for d in wmem))
    set_h = set((tuple(sorted(d.items())) for d in hmem))
    intersection = set_w.intersection(set_h)
    for element in intersection:
        intersect.append(dict(((x, y) for x, y in element)))
    return intersect