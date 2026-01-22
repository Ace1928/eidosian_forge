from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver, zapis_svm
def rename_vserver(self):
    """ ZAPI only, for REST it is handled as a modify"""
    vserver_rename = netapp_utils.zapi.NaElement.create_node_with_children('vserver-rename', **{'vserver-name': self.parameters['from_name'], 'new-name': self.parameters['name']})
    try:
        self.server.invoke_successfully(vserver_rename, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as exc:
        self.module.fail_json(msg='Error renaming SVM %s: %s' % (self.parameters['from_name'], to_native(exc)), exception=traceback.format_exc())