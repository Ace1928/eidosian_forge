from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.frr.frr.plugins.module_utils.network.frr.frr import (
def populate_mpls_ldp_neighbors(self, data):
    facts = {}
    entries = data.splitlines()
    for x in entries:
        if x.startswith('AF'):
            continue
        x = x.split()
        if len(x) > 0:
            ldp = {}
            ldp['neighbor'] = x[1]
            ldp['source'] = x[3]
            facts[ldp['source']] = []
            facts[ldp['source']].append(ldp)
    return facts