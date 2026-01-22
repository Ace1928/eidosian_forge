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
def create_volume_options(self):
    """Set volume options for create operation"""
    options = {}
    if self.volume_style == 'flexgroup':
        options['volume-name'] = self.parameters['name']
        if self.parameters.get('aggr_list_multiplier') is not None:
            options['aggr-list-multiplier'] = str(self.parameters['aggr_list_multiplier'])
        if self.parameters.get('auto_provision_as') is not None:
            options['auto-provision-as'] = self.parameters['auto_provision_as']
        if self.parameters.get('space_guarantee') is not None:
            options['space-guarantee'] = self.parameters['space_guarantee']
    else:
        options['volume'] = self.parameters['name']
        if self.parameters.get('aggregate_name') is None:
            self.module.fail_json(msg='Error provisioning volume %s: aggregate_name is required' % self.parameters['name'])
        options['containing-aggr-name'] = self.parameters['aggregate_name']
        if self.parameters.get('space_guarantee') is not None:
            options['space-reserve'] = self.parameters['space_guarantee']
    if self.parameters.get('size') is not None:
        options['size'] = str(self.parameters['size'])
    if self.parameters.get('snapshot_policy') is not None:
        options['snapshot-policy'] = self.parameters['snapshot_policy']
    if self.parameters.get('unix_permissions') is not None:
        options['unix-permissions'] = self.parameters['unix_permissions']
    if self.parameters.get('group_id') is not None:
        options['group-id'] = str(self.parameters['group_id'])
    if self.parameters.get('user_id') is not None:
        options['user-id'] = str(self.parameters['user_id'])
    if self.parameters.get('volume_security_style') is not None:
        options['volume-security-style'] = self.parameters['volume_security_style']
    if self.parameters.get('export_policy') is not None:
        options['export-policy'] = self.parameters['export_policy']
    if self.parameters.get('junction_path') is not None:
        options['junction-path'] = self.parameters['junction_path']
    if self.parameters.get('comment') is not None:
        options['volume-comment'] = self.parameters['comment']
    if self.parameters.get('type') is not None:
        options['volume-type'] = self.parameters['type']
    if self.parameters.get('percent_snapshot_space') is not None:
        options['percentage-snapshot-reserve'] = str(self.parameters['percent_snapshot_space'])
    if self.parameters.get('language') is not None:
        options['language-code'] = self.parameters['language']
    if self.parameters.get('qos_policy_group') is not None:
        options['qos-policy-group-name'] = self.parameters['qos_policy_group']
    if self.parameters.get('qos_adaptive_policy_group') is not None:
        options['qos-adaptive-policy-group-name'] = self.parameters['qos_adaptive_policy_group']
    if self.parameters.get('nvfail_enabled') is not None:
        options['is-nvfail-enabled'] = str(self.parameters['nvfail_enabled'])
    if self.parameters.get('space_slo') is not None:
        options['space-slo'] = self.parameters['space_slo']
    if self.parameters.get('tiering_policy') is not None:
        options['tiering-policy'] = self.parameters['tiering_policy']
    if self.parameters.get('encrypt') is not None:
        options['encrypt'] = self.na_helper.get_value_for_bool(False, self.parameters['encrypt'], 'encrypt')
    if self.parameters.get('vserver_dr_protection') is not None:
        options['vserver-dr-protection'] = self.parameters['vserver_dr_protection']
    if self.parameters['is_online']:
        options['volume-state'] = 'online'
    else:
        options['volume-state'] = 'offline'
    return options