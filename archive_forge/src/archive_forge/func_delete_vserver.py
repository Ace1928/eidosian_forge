from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver, zapis_svm
def delete_vserver(self, current=None):
    if self.use_rest:
        if current is None:
            self.module.fail_json(msg='Internal error, expecting SVM object in delete')
        dummy, error = rest_generic.delete_async(self.rest_api, 'svm/svms', current['uuid'], timeout=self.timeout)
        if error:
            self.module.fail_json(msg='Error in delete: %s' % error)
    else:
        vserver_delete = netapp_utils.zapi.NaElement.create_node_with_children('vserver-destroy', **{'vserver-name': self.parameters['name']})
        try:
            self.server.invoke_successfully(vserver_delete, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as exc:
            self.module.fail_json(msg='Error deleting SVM %s: %s' % (self.parameters['name'], to_native(exc)), exception=traceback.format_exc())