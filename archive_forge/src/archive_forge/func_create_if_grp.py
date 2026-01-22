from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_if_grp(self):
    """
        Creates a new ifgrp
        """
    if self.use_rest:
        return self.create_if_grp_rest()
    route_obj = netapp_utils.zapi.NaElement('net-port-ifgrp-create')
    route_obj.add_new_child('distribution-function', self.parameters['distribution_function'])
    route_obj.add_new_child('ifgrp-name', self.parameters['name'])
    route_obj.add_new_child('mode', self.parameters['mode'])
    route_obj.add_new_child('node', self.parameters['node'])
    try:
        self.server.invoke_successfully(route_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating if_group %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    if self.parameters.get('ports') is not None:
        for port in self.parameters.get('ports'):
            self.add_port_to_if_grp(port)