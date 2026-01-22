from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_iptonetwork_mappings(self):
    network_mappings = self.module.params.get('ip_to_networks')
    if network_mappings is None:
        return
    if network_mappings and self.module.params.get('networks'):
        self.module.fail_json(msg='networks and ip_to_networks are mutually exclusive.')
    network_names = [n['network'] for n in network_mappings]
    ids = self.get_network_ids(network_names)
    res = []
    for i, data in enumerate(network_mappings):
        res.append(dict(networkid=ids[i], **data))
    return res