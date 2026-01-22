from __future__ import absolute_import, division, print_function
import os
import uuid
from ansible.module_utils.basic import AnsibleModule
def serialize_sshkey(sshkey):
    sshkey_data = {}
    copy_keys = ['id', 'key', 'label', 'fingerprint']
    for name in copy_keys:
        sshkey_data[name] = getattr(sshkey, name)
    return sshkey_data