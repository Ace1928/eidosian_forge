from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_zone(self):
    self.log('Lists all record sets in a DNS zone')
    try:
        response = self.dns_client.record_sets.list_by_dns_zone(self.resource_group, self.zone_name, top=self.top)
    except Exception as exc:
        self.fail('Failed to list for zone {0} - {1}'.format(self.zone_name, str(exc)))
    results = []
    for item in response:
        results.append(item)
    return results