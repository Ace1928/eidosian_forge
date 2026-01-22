from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.api import (
def parse_detail(self, data):
    for entry in data:
        if 'interface' not in entry:
            continue
        entry.pop('.id', None)
        yield entry