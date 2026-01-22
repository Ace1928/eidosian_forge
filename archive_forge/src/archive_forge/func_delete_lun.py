from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_volume
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_lun(self, path):
    """
        Delete requested LUN
        """
    if self.use_rest:
        return self.delete_lun_rest()
    lun_delete = netapp_utils.zapi.NaElement.create_node_with_children('lun-destroy', **{'path': path, 'force': str(self.parameters['force_remove']), 'destroy-fenced-lun': str(self.parameters['force_remove_fenced'])})
    try:
        self.server.invoke_successfully(lun_delete, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as exc:
        self.module.fail_json(msg='Error deleting lun %s: %s' % (path, to_native(exc)), exception=traceback.format_exc())