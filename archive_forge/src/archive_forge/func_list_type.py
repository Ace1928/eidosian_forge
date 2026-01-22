from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def list_type(self):
    self.log('Lists the record sets of a specified type in a DNS zone')
    try:
        response = self.dns_client.record_sets.list_by_type(self.resource_group, self.zone_name, self.record_type, top=self.top)
    except Exception as exc:
        self.fail('Failed to list for record type {0} - {1}'.format(self.record_type, str(exc)))
    results = []
    for item in response:
        results.append(item)
    return results