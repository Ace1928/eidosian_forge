from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def delete_name_service_switch(self):
    """
        delete name service switch
        :return: None
        """
    nss_delete = netapp_utils.zapi.NaElement.create_node_with_children('nameservice-nsswitch-destroy', **{'nameservice-database': self.parameters['database_type']})
    try:
        self.server.invoke_successfully(nss_delete, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error on deleting name service switch config on vserver %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())