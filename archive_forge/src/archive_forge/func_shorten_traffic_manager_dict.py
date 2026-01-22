from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, normalize_location_name
def shorten_traffic_manager_dict(tmd):
    return dict(id=tmd['id'], endpoints=[endpoint['id'] for endpoint in tmd['endpoints']] if tmd['endpoints'] else [])