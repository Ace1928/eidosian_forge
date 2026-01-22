from __future__ import absolute_import, division, print_function
import re
import shlex
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.parsing import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_lines
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
def parse_rpcs(module):
    items = list()
    for rpc in module.params['rpcs'] or list():
        parts = shlex.split(rpc)
        name = parts.pop(0)
        args = dict()
        for item in parts:
            key, value = item.split('=')
            if str(value).upper() in ['TRUE', 'FALSE']:
                args[key] = bool(value)
            elif re.match('^[0-9]+$', value):
                args[key] = int(value)
            else:
                args[key] = str(value)
        display = module.params['display'] or 'xml'
        if display == 'set' and rpc != 'get-configuration':
            module.fail_json(msg="Invalid display option '%s' given for rpc '%s'" % ('set', name))
        xattrs = {'format': display}
        items.append({'name': name, 'args': args, 'xattrs': xattrs})
    return items