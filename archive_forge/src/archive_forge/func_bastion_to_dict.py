from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def bastion_to_dict(self, bastion_info):
    bastion = bastion_info.as_dict()
    result = dict(id=bastion.get('id'), name=bastion.get('name'), type=bastion.get('type'), etag=bastion.get('etag'), location=bastion.get('location'), tags=bastion.get('tags'), sku=dict(), ip_configurations=list(), dns_name=bastion.get('dns_name'), provisioning_state=bastion.get('provisioning_state'), scale_units=bastion.get('scale_units'), disable_copy_paste=bastion.get('disable_copy_paste'), enable_file_copy=bastion.get('enable_file_copy'), enable_ip_connect=bastion.get('enable_ip_connect'), enable_shareable_link=bastion.get('enable_tunneling'), enable_tunneling=bastion.get('enable_tunneling'))
    if bastion.get('sku'):
        result['sku']['name'] = bastion['sku']['name']
    if bastion.get('ip_configurations'):
        for items in bastion['ip_configurations']:
            result['ip_configurations'].append({'name': items['name'], 'subnet': dict(id=items['subnet']['id']), 'public_ip_address': dict(id=items['public_ip_address']['id']), 'private_ip_allocation_method': items['private_ip_allocation_method']})
    return result