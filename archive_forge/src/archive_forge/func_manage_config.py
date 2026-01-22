from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native, to_bytes
from hashlib import sha1
def manage_config(self, cursor, state):
    if state:
        if self.save_to_disk:
            save_config_to_disk(cursor, 'USERS')
        if self.load_to_runtime:
            load_config_to_runtime(cursor, 'USERS')