from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
@_api_permission_denied_handler('api_clients')
def generate_api_client_dict(module, fusion):
    client_info = {}
    api_instance = purefusion.IdentityManagerApi(fusion)
    clients = api_instance.list_api_clients()
    for client in clients:
        client_info[client.name] = {'display_name': client.display_name, 'issuer': client.issuer, 'public_key': client.public_key, 'creator_id': client.creator_id, 'last_key_update': time.strftime('%a, %d %b %Y %H:%M:%S %Z', time.localtime(client.last_key_update / 1000)), 'last_used': time.strftime('%a, %d %b %Y %H:%M:%S %Z', time.localtime(client.last_used / 1000))}
    return client_info