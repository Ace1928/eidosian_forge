from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def normalize_proposed_values(proposed, module):
    keys = proposed.keys()
    if 'bfd' in keys:
        proposed['bfd'] = proposed['bfd'].lower()
    if 'hello_interval' in keys:
        hello_interval = proposed['hello_interval']
        if not module.params['hello_interval_ms']:
            hello_interval = hello_interval * 1000
        proposed['hello_interval'] = str(hello_interval)