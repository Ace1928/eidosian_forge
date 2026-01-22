from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_address_info(self, address_ranges):
    if address_ranges is None:
        return None
    address_info = []
    for address in address_ranges:
        address_range = {}
        if '-' in address:
            address_range['end'] = address.split('-')[1]
            address_range['start'] = address.split('-')[0]
        else:
            address_range['end'] = address
            address_range['start'] = address
        address_info.append(address_range)
    return address_info