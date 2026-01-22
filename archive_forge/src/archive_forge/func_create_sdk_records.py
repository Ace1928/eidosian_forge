from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, HAS_AZURE
def create_sdk_records(self, input_records, record_type):
    record = RECORDSET_VALUE_MAP.get(record_type)
    if not record:
        self.fail('record type {0} is not supported now'.format(record_type))
    record_sdk_class = getattr(self.dns_models, record.get('classobj'))
    return [record_sdk_class(**x) for x in input_records]