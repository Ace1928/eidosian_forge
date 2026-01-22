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
def start_encryption_conversion(self, encrypt_destination):
    if encrypt_destination:
        if self.use_rest:
            return self.encryption_conversion_rest()
        zapi = netapp_utils.zapi.NaElement.create_node_with_children('volume-encryption-conversion-start', **{'volume': self.parameters['name']})
        try:
            self.server.invoke_successfully(zapi, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error enabling encryption for volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        if self.parameters.get('wait_for_completion'):
            self.wait_for_volume_encryption_conversion()
    else:
        self.module.warn('disabling encryption requires cluster admin permissions.')
        self.move_volume(encrypt_destination)