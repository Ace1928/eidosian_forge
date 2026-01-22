from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import to_native
from ansible_collections.netapp.azure.plugins.module_utils.azure_rm_netapp_common import AzureRMNetAppModuleBase
from ansible_collections.netapp.azure.plugins.module_utils.netapp_module import NetAppModule
def get_azure_netapp_account(self):
    """
            Returns NetApp Account object for an existing account
            Return None if account does not exist
        """
    try:
        account_get = self.netapp_client.accounts.get(self.parameters['resource_group'], self.parameters['name'])
    except (CloudError, ResourceNotFoundError):
        return None
    account = vars(account_get)
    ads = None
    if account.get('active_directories') is not None:
        ads = list()
        for each_ad in account.get('active_directories'):
            ad_dict = vars(each_ad)
            dns = ad_dict.get('dns')
            if dns is not None:
                ad_dict['dns'] = sorted(dns.split(','))
            ads.append(ad_dict)
    account['active_directories'] = ads
    return account