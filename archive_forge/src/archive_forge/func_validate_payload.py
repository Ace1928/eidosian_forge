from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
import json
def validate_payload(meraki):
    for peer in meraki.params['peers']:
        if peer['name'] is None:
            meraki.fail_json(msg='Peer name must be specified')
        elif peer['public_ip'] is None:
            meraki.fail_json(msg='Peer public IP must be specified')
        elif peer['secret'] is None:
            meraki.fail_json(msg='Peer secret must be specified')
        elif peer['private_subnets'] is None:
            meraki.fail_json(msg='Peer private subnets must be specified')