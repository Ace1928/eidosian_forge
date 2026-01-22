from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def build_body_for_create(self):
    body = {'name': self.parameters['policy_name']}
    if self.parameters.get('vserver') is not None:
        body['svm'] = {'name': self.parameters['vserver']}
    policy_type = 'async'
    if 'policy_type' in self.parameters:
        if 'async' in self.parameters['policy_type']:
            policy_type = 'async'
        elif 'sync' in self.parameters['policy_type']:
            policy_type = 'sync'
            body['sync_type'] = 'sync'
            if 'sync_type' in self.parameters:
                body['sync_type'] = self.parameters['sync_type']
        body['type'] = policy_type
    if 'copy_all_source_snapshots' in self.parameters:
        body['copy_all_source_snapshots'] = self.parameters['copy_all_source_snapshots']
    if 'copy_latest_source_snapshot' in self.parameters:
        body['copy_latest_source_snapshot'] = self.parameters['copy_latest_source_snapshot']
    if 'create_snapshot_on_source' in self.parameters:
        snapmirror_policy_retention_objs = []
        for index, rule in enumerate(self.parameters['snapmirror_label']):
            retention = {'label': rule, 'count': str(self.parameters['keep'][index])}
            if 'prefix' in self.parameters and self.parameters['prefix'] != '':
                retention['prefix'] = self.parameters['prefix'][index]
            if 'schedule' in self.parameters and self.parameters['schedule'] != '':
                retention['creation_schedule'] = {'name': self.parameters['schedule'][index]}
            snapmirror_policy_retention_objs.append(retention)
        body['retention'] = snapmirror_policy_retention_objs
        body['create_snapshot_on_source'] = self.parameters['create_snapshot_on_source']
    return self.build_body_for_create_or_modify(policy_type, body)