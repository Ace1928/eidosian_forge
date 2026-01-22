from __future__ import absolute_import, division, print_function
from copy import deepcopy
from functools import partial
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
def handle_purge(module, want):
    want_users = [item['name'] for item in want]
    element = Element('system')
    login = SubElement(element, 'login')
    conn = get_connection(module)
    try:
        reply = conn.execute_rpc(tostring(Element('get-configuration')), ignore_warning=False)
    except ConnectionError as exc:
        module.fail_json(msg=to_text(exc, errors='surrogate_then_replace'))
    users = reply.xpath('configuration/system/login/user/name')
    if users:
        for item in users:
            name = item.text
            if name not in want_users and name != 'root':
                user = SubElement(login, 'user', {'operation': 'delete'})
                SubElement(user, 'name').text = name
    if element.xpath('/system/login/user/name'):
        return element