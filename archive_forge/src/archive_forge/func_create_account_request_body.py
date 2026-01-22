from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import to_native
from ansible_collections.netapp.azure.plugins.module_utils.azure_rm_netapp_common import AzureRMNetAppModuleBase
from ansible_collections.netapp.azure.plugins.module_utils.netapp_module import NetAppModule
def create_account_request_body(self, modify=None):
    """
            Create an Azure NetApp Account Request Body
            :return: None
        """
    options = dict()
    location = None
    for attr in ('location', 'tags', 'active_directories'):
        value = self.parameters.get(attr)
        if attr == 'location' and modify is None:
            location = value
            continue
        if value is not None:
            if modify is None or attr in modify:
                if attr == 'active_directories':
                    ads = list()
                    for ad_dict in value:
                        if ad_dict.get('dns') is not None:
                            ad_dict['dns'] = ','.join(ad_dict['dns'])
                        ads.append(ActiveDirectory(**self.na_helper.filter_out_none_entries(ad_dict)))
                    value = ads
                options[attr] = value
    if modify is None:
        if location is None:
            self.module.fail_json(msg="Error: 'location' is a required parameter")
        return NetAppAccount(location=location, **options)
    return NetAppAccountPatch(**options)