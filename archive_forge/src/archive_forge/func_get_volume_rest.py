from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def get_volume_rest(self, vol_name):
    """
        This covers the zapi functions
        get_volume
         - volume_get_iter
         - get_efficiency_info
        """
    api = 'storage/volumes'
    params = {'name': vol_name, 'svm.name': self.parameters['vserver'], 'fields': 'encryption.enabled,tiering.policy,nas.export_policy.name,aggregates.name,aggregates.uuid,uuid,nas.path,style,type,comment,qos.policy.name,nas.security_style,nas.gid,nas.unix_permissions,nas.uid,snapshot_policy,space.snapshot.reserve_percent,space.size,guarantee.type,state,efficiency.compression,snaplock,files.maximum,space.logical_space.enforcement,space.logical_space.reporting,'}
    if self.parameters.get('efficiency_policy'):
        params['fields'] += 'efficiency.policy.name,'
    if self.parameters.get('tiering_minimum_cooling_days'):
        params['fields'] += 'tiering.min_cooling_days,'
    if self.parameters.get('analytics'):
        params['fields'] += 'analytics,'
    if self.parameters.get('tags'):
        params['fields'] += '_tags,'
    if self.parameters.get('atime_update') is not None:
        params['fields'] += 'access_time_enabled,'
    if self.parameters.get('snapdir_access') is not None:
        params['fields'] += 'snapshot_directory_access_enabled,'
    if self.parameters.get('snapshot_auto_delete') is not None:
        params['fields'] += 'space.snapshot.autodelete,'
    if self.parameters.get('vol_nearly_full_threshold_percent') is not None:
        params['fields'] += 'space.nearly_full_threshold_percent,'
    if self.parameters.get('vol_full_threshold_percent') is not None:
        params['fields'] += 'space.full_threshold_percent,'
    record, error = rest_generic.get_one_record(self.rest_api, api, params)
    if error:
        self.module.fail_json(msg=error)
    return self.format_get_volume_rest(record) if record else None