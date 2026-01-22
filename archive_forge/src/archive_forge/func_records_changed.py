from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, HAS_AZURE
def records_changed(self, input_records, server_records):
    if not isinstance(server_records, list):
        server_records = [server_records]
    input_set = set([self.module.jsonify(x.as_dict()) for x in input_records])
    server_set = set([self.module.jsonify(x.as_dict()) for x in server_records])
    if self.record_mode == 'append':
        input_set = server_set.union(input_set)
    changed = input_set != server_set
    records = [self.module.from_json(x) for x in input_set]
    return (self.create_sdk_records(records, self.record_type), changed)