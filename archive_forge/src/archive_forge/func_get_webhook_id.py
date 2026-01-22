from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
def get_webhook_id(name, webhooks):
    for webhook in webhooks:
        if name == webhook['name']:
            return webhook['id']
    return None