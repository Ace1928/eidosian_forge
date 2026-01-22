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
def build_zapi_volume_modify_iter(self, params):
    vol_mod_iter = netapp_utils.zapi.NaElement('volume-modify-iter-async' if self.volume_style == 'flexgroup' or self.parameters['is_infinite'] else 'volume-modify-iter')
    attributes = netapp_utils.zapi.NaElement('attributes')
    vol_mod_attributes = netapp_utils.zapi.NaElement('volume-attributes')
    vol_inode_attributes = netapp_utils.zapi.NaElement('volume-inode-attributes')
    self.create_volume_attribute(vol_inode_attributes, vol_mod_attributes, 'files-total', 'max_files', int)
    vol_space_attributes = netapp_utils.zapi.NaElement('volume-space-attributes')
    self.create_volume_attribute(vol_space_attributes, vol_mod_attributes, 'space-guarantee', 'space_guarantee')
    self.create_volume_attribute(vol_space_attributes, vol_mod_attributes, 'percentage-snapshot-reserve', 'percent_snapshot_space', int)
    self.create_volume_attribute(vol_space_attributes, vol_mod_attributes, 'space-slo', 'space_slo')
    vol_snapshot_attributes = netapp_utils.zapi.NaElement('volume-snapshot-attributes')
    self.create_volume_attribute(vol_snapshot_attributes, vol_mod_attributes, 'snapshot-policy', 'snapshot_policy')
    self.create_volume_attribute(vol_snapshot_attributes, vol_mod_attributes, 'snapdir-access-enabled', 'snapdir_access', bool)
    self.create_volume_attribute('volume-export-attributes', vol_mod_attributes, 'policy', 'export_policy')
    if self.parameters.get('unix_permissions') is not None or self.parameters.get('group_id') is not None or self.parameters.get('user_id') is not None:
        vol_security_attributes = netapp_utils.zapi.NaElement('volume-security-attributes')
        vol_security_unix_attributes = netapp_utils.zapi.NaElement('volume-security-unix-attributes')
        self.create_volume_attribute(vol_security_unix_attributes, vol_security_attributes, 'permissions', 'unix_permissions')
        self.create_volume_attribute(vol_security_unix_attributes, vol_security_attributes, 'group-id', 'group_id', int)
        self.create_volume_attribute(vol_security_unix_attributes, vol_security_attributes, 'user-id', 'user_id', int)
        vol_mod_attributes.add_child_elem(vol_security_attributes)
    if params and params.get('volume_security_style') is not None:
        self.create_volume_attribute('volume-security-attributes', vol_mod_attributes, 'style', 'volume_security_style')
    self.create_volume_attribute('volume-performance-attributes', vol_mod_attributes, 'is-atime-update-enabled', 'atime_update', bool)
    self.create_volume_attribute('volume-qos-attributes', vol_mod_attributes, 'policy-group-name', 'qos_policy_group')
    self.create_volume_attribute('volume-qos-attributes', vol_mod_attributes, 'adaptive-policy-group-name', 'qos_adaptive_policy_group')
    if params and params.get('tiering_policy') is not None:
        self.create_volume_attribute('volume-comp-aggr-attributes', vol_mod_attributes, 'tiering-policy', 'tiering_policy')
    self.create_volume_attribute('volume-state-attributes', vol_mod_attributes, 'is-nvfail-enabled', 'nvfail_enabled', bool)
    self.create_volume_attribute('volume-vserver-dr-protection-attributes', vol_mod_attributes, 'vserver-dr-protection', 'vserver_dr_protection')
    self.create_volume_attribute('volume-id-attributes', vol_mod_attributes, 'comment', 'comment')
    attributes.add_child_elem(vol_mod_attributes)
    query = netapp_utils.zapi.NaElement('query')
    vol_query_attributes = netapp_utils.zapi.NaElement('volume-attributes')
    self.create_volume_attribute('volume-id-attributes', vol_query_attributes, 'name', 'name')
    query.add_child_elem(vol_query_attributes)
    vol_mod_iter.add_child_elem(attributes)
    vol_mod_iter.add_child_elem(query)
    return vol_mod_iter