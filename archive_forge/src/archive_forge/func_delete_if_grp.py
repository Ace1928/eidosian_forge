from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_if_grp(self, uuid=None):
    """
        Deletes a ifgrp
        """
    if self.use_rest:
        api = 'network/ethernet/ports'
        dummy, error = rest_generic.delete_async(self.rest_api, api, uuid)
        if error:
            self.module.fail_json(msg=error)
    else:
        route_obj = netapp_utils.zapi.NaElement('net-port-ifgrp-destroy')
        route_obj.add_new_child('ifgrp-name', self.parameters['name'])
        route_obj.add_new_child('node', self.parameters['node'])
        try:
            self.server.invoke_successfully(route_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error deleting if_group %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())