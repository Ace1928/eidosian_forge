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
def format_get_volume_rest(self, record):
    is_online = record.get('state') == 'online'
    aggregates = record.get('aggregates', None)
    aggr_name = aggregates[0].get('name', None) if aggregates else None
    rest_compression = self.na_helper.safe_get(record, ['efficiency', 'compression'])
    junction_path = self.na_helper.safe_get(record, ['nas', 'path'])
    if junction_path is None:
        junction_path = ''
    state = self.na_helper.safe_get(record, ['analytics', 'state'])
    analytics = 'on' if state == 'initializing' else state
    auto_delete_info = self.na_helper.safe_get(record, ['space', 'snapshot', 'autodelete'])
    if auto_delete_info is not None:
        auto_delete_info['state'] = self.enabled_to_bool(self.na_helper.safe_get(record, ['space', 'snapshot', 'autodelete', 'enabled']), reverse=True)
        del auto_delete_info['enabled']
    return {'tags': record.get('_tags', []), 'name': record.get('name', None), 'analytics': analytics, 'encrypt': self.na_helper.safe_get(record, ['encryption', 'enabled']), 'tiering_policy': self.na_helper.safe_get(record, ['tiering', 'policy']), 'export_policy': self.na_helper.safe_get(record, ['nas', 'export_policy', 'name']), 'aggregate_name': aggr_name, 'aggregates': aggregates, 'flexgroup_uuid': record.get('uuid', None), 'instance_uuid': record.get('uuid', None), 'junction_path': junction_path, 'style_extended': record.get('style', None), 'type': record.get('type', None), 'comment': record.get('comment', None), 'qos_policy_group': self.na_helper.safe_get(record, ['qos', 'policy', 'name']), 'qos_adaptive_policy_group': self.na_helper.safe_get(record, ['qos', 'policy', 'name']), 'volume_security_style': self.na_helper.safe_get(record, ['nas', 'security_style']), 'group_id': self.na_helper.safe_get(record, ['nas', 'gid']), 'unix_permissions': str(self.na_helper.safe_get(record, ['nas', 'unix_permissions'])), 'user_id': self.na_helper.safe_get(record, ['nas', 'uid']), 'snapshot_policy': self.na_helper.safe_get(record, ['snapshot_policy', 'name']), 'percent_snapshot_space': self.na_helper.safe_get(record, ['space', 'snapshot', 'reserve_percent']), 'size': self.na_helper.safe_get(record, ['space', 'size']), 'space_guarantee': self.na_helper.safe_get(record, ['guarantee', 'type']), 'is_online': is_online, 'uuid': record.get('uuid', None), 'efficiency_policy': self.na_helper.safe_get(record, ['efficiency', 'policy', 'name']), 'compression': rest_compression in ('both', 'background'), 'inline_compression': rest_compression in ('both', 'inline'), 'logical_space_enforcement': self.na_helper.safe_get(record, ['space', 'logical_space', 'enforcement']), 'logical_space_reporting': self.na_helper.safe_get(record, ['space', 'logical_space', 'reporting']), 'tiering_minimum_cooling_days': self.na_helper.safe_get(record, ['tiering', 'min_cooling_days']), 'snaplock': self.na_helper.safe_get(record, ['snaplock']), 'max_files': self.na_helper.safe_get(record, ['files', 'maximum']), 'atime_update': record.get('access_time_enabled', True), 'snapdir_access': record.get('snapshot_directory_access_enabled', True), 'snapshot_auto_delete': auto_delete_info, 'vol_nearly_full_threshold_percent': self.na_helper.safe_get(record, ['space', 'nearly_full_threshold_percent']), 'vol_full_threshold_percent': self.na_helper.safe_get(record, ['space', 'full_threshold_percent'])}