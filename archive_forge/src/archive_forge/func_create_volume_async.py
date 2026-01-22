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
def create_volume_async(self):
    """
        create volume async.
        """
    options = self.create_volume_options()
    volume_create = netapp_utils.zapi.NaElement.create_node_with_children('volume-create-async', **options)
    if self.parameters.get('aggr_list'):
        aggr_list_obj = netapp_utils.zapi.NaElement('aggr-list')
        volume_create.add_child_elem(aggr_list_obj)
        for aggr in self.parameters['aggr_list']:
            aggr_list_obj.add_new_child('aggr-name', aggr)
    try:
        result = self.server.invoke_successfully(volume_create, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        size_msg = ' of size %s' % self.parameters['size'] if self.parameters.get('size') is not None else ''
        self.module.fail_json(msg='Error provisioning volume %s%s: %s' % (self.parameters['name'], size_msg, to_native(error)), exception=traceback.format_exc())
    self.check_invoke_result(result, 'create')
    return None