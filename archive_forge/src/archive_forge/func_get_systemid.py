from __future__ import absolute_import, division, print_function
import ssl
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import xmlrpc_client
def get_systemid(client, session, sysname):
    systems = client.system.listUserSystems(session)
    for system in systems:
        if system.get('name') == sysname:
            idres = system.get('id')
            idd = int(idres)
            return idd