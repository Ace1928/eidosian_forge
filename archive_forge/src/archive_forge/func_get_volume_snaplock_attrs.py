from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def get_volume_snaplock_attrs(self):
    """
        Return volume-get-snaplock-attrs query results
        :param vol_name: name of the volume
        :return: dict of the volume snaplock attrs
        """
    volume_snaplock = netapp_utils.zapi.NaElement('volume-get-snaplock-attrs')
    volume_snaplock.add_new_child('volume', self.parameters['name'])
    try:
        result = self.server.invoke_successfully(volume_snaplock, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching snaplock attributes for volume %s : %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    return_value = None
    if result.get_child_by_name('snaplock-attrs'):
        volume_snaplock_attributes = result['snaplock-attrs']['snaplock-attrs-info']
        return_value = {'autocommit_period': volume_snaplock_attributes['autocommit-period'], 'default_retention_period': volume_snaplock_attributes['default-retention-period'], 'is_volume_append_mode_enabled': self.na_helper.get_value_for_bool(True, volume_snaplock_attributes['is-volume-append-mode-enabled']), 'maximum_retention_period': volume_snaplock_attributes['maximum-retention-period'], 'minimum_retention_period': volume_snaplock_attributes['minimum-retention-period']}
    return return_value