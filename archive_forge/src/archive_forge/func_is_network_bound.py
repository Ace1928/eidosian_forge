from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def is_network_bound(meraki, nets, net_id, template_id):
    for net in nets:
        if net['id'] == net_id:
            try:
                if net['configTemplateId'] == template_id:
                    return True
            except KeyError:
                pass
    return False