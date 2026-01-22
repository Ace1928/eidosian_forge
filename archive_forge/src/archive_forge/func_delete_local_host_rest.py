from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def delete_local_host_rest(self):
    """
        vserver services name-service dns hosts delete.
        """
    api = 'name-services/local-hosts/%s/%s' % (self.owner_uuid, self.parameters['address'])
    dummy, error = rest_generic.delete_async(self.rest_api, api, None)
    if error:
        self.module.fail_json(msg='Error deleting IP to hostname mappings for %s: %s' % (self.parameters['owner'], to_native(error)), exception=traceback.format_exc())