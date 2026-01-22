from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.storage_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def partition_probe(self, data):
    if self.replicationpolicy and self.noreplicationpolicy:
        self.module.fail_json(msg='Mutual exclusive parameters: {0}, {1}'.format('replicationpolicy', 'noreplicationpolicy'))
    if self.replicationpolicy and self.preferredmanagementsystem:
        self.module.fail_json(msg='Mutual exclusive parameters: {0}, {1}'.format('replicationpolicy', 'preferredmanagementsystem'))
    if self.deletepreferredmanagementcopy and (not self.noreplicationpolicy):
        self.module.fail_json(msg='These parameters must be passed together: {0}, {1}'.format('deletepreferredmanagementcopy', 'noreplicationpolicy'))
    params_mapping = (('replicationpolicy', data.get('replication_policy_name', '')), ('preferredmanagementsystem', data.get('preferred_management_system_name', '')), ('noreplicationpolicy', not bool(data.get('replication_policy_name', ''))))
    props = dict(((k, getattr(self, k)) for k, v in params_mapping if getattr(self, k) and getattr(self, k) != v))
    if self.noreplicationpolicy in props:
        if self.deletepreferredmanagementcopy:
            props.append('deletepreferredmanagementcopy')
    self.log('Storage Partition props = %s', props)
    return props