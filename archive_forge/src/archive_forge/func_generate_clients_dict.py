from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_clients_dict(array):
    clients_info = {}
    clients = list(array.get_api_clients().items)
    for client in range(0, len(clients)):
        c_name = clients[client].name
        clients_info[c_name] = {'enabled': clients[client].enabled, 'TTL(seconds)': clients[client].access_token_ttl_in_ms / 1000, 'key_id': clients[client].key_id, 'client_id': clients[client].id, 'max_role': clients[client].max_role, 'public_key': clients[client].public_key}
    return clients_info