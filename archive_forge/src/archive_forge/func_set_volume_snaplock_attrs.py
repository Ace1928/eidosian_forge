from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def set_volume_snaplock_attrs(self, modify):
    """Set ONTAP volume snaplock attributes"""
    volume_snaplock_obj = netapp_utils.zapi.NaElement('volume-set-snaplock-attrs')
    volume_snaplock_obj.add_new_child('volume', self.parameters['name'])
    if modify.get('autocommit_period') is not None:
        volume_snaplock_obj.add_new_child('autocommit-period', self.parameters['autocommit_period'])
    if modify.get('default_retention_period') is not None:
        volume_snaplock_obj.add_new_child('default-retention-period', self.parameters['default_retention_period'])
    if modify.get('is_volume_append_mode_enabled') is not None:
        volume_snaplock_obj.add_new_child('is-volume-append-mode-enabled', self.na_helper.get_value_for_bool(False, self.parameters['is_volume_append_mode_enabled']))
    if modify.get('maximum_retention_period') is not None:
        volume_snaplock_obj.add_new_child('maximum-retention-period', self.parameters['maximum_retention_period'])
    if modify.get('minimum_retention_period') is not None:
        volume_snaplock_obj.add_new_child('minimum-retention-period', self.parameters['minimum_retention_period'])
    try:
        self.server.invoke_successfully(volume_snaplock_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error setting snaplock attributes for volume %s : %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())