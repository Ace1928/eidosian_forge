from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def create_name_service_switch(self):
    """
        create name service switch config
        :return: None
        """
    nss_create = netapp_utils.zapi.NaElement('nameservice-nsswitch-create')
    nss_create.add_new_child('nameservice-database', self.parameters['database_type'])
    nss_sources = netapp_utils.zapi.NaElement('nameservice-sources')
    nss_create.add_child_elem(nss_sources)
    for source in self.parameters['sources']:
        nss_sources.add_new_child('nss-source-type', source)
    try:
        self.server.invoke_successfully(nss_create, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error on creating name service switch config on vserver %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())