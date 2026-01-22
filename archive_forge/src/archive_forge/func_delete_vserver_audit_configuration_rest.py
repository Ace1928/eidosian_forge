from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_vserver_audit_configuration_rest(self, current):
    """
        Deletes an audit configuration.
        """
    api = 'protocols/audit/%s' % self.svm_uuid
    if current['enabled'] is True:
        modify = {'enabled': False}
        self.modify_vserver_audit_configuration_rest(modify)
        current = self.get_vserver_audit_configuration_rest()
    retry = 2
    while retry > 0:
        record, error = rest_generic.delete_async(self.rest_api, api, None)
        if error and '9699350' in error:
            time.sleep(120)
            retry -= 1
        elif error:
            self.module.fail_json(msg='Error on deleting vserver audit configuration: %s' % error)
        else:
            return