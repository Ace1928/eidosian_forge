from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def rename_interface(self):
    options = {'interface-name': self.parameters['from_name'], 'new-name': self.parameters['interface_name'], 'vserver': self.parameters['vserver']}
    interface_rename = netapp_utils.zapi.NaElement.create_node_with_children('net-interface-rename', **options)
    try:
        self.server.invoke_successfully(interface_rename, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error renaming %s to %s: %s' % (self.parameters['from_name'], self.parameters['interface_name'], to_native(error)), exception=traceback.format_exc())