from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def modify_interface_rest(self, uuid, body):
    """ calling REST to modify interface """
    if not body:
        return
    dummy, error = rest_generic.patch_async(self.rest_api, self.get_net_int_api(), uuid, body)
    if error:
        self.module.fail_json(msg='Error modifying interface %s: %s' % (self.parameters['interface_name'], to_native(error)), exception=traceback.format_exc())