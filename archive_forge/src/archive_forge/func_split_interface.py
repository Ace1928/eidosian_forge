from __future__ import absolute_import, division, print_function
import copy
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def split_interface(interface):
    match = re.search('(\\D+)(\\S*)', interface, re.M)
    if match:
        return (match.group(1), match.group(2))