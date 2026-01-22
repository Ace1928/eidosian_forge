from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.arista.eos.plugins.module_utils.network.eos.eos import (
def collect_facts(module, result):
    out = run_commands(module, ['show management api http-commands | json'])
    facts = dict(eos_eapi_urls=dict())
    for each in out[0]['urls']:
        intf, url = each.split(':', 1)
        key = str(intf).strip()
        if key not in facts['eos_eapi_urls']:
            facts['eos_eapi_urls'][key] = list()
        facts['eos_eapi_urls'][key].append(str(url).strip())
    result['ansible_facts'] = facts