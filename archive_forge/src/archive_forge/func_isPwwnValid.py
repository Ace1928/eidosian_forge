from __future__ import absolute_import, division, print_function
import string
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def isPwwnValid(pwwn):
    pwwnsplit = pwwn.split(':')
    if len(pwwnsplit) != 8:
        return False
    for eachpwwnsplit in pwwnsplit:
        if len(eachpwwnsplit) > 2 or len(eachpwwnsplit) < 1:
            return False
        if not all((c in string.hexdigits for c in eachpwwnsplit)):
            return False
    return True