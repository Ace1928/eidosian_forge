from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_vlan_info(self):
    """
        Create a vlan_info object to be used in a create/delete
        :return:
        """
    vlan_info = netapp_utils.zapi.NaElement('vlan-info')
    vlan_info.add_new_child('parent-interface', self.parameters['parent_interface'])
    vlan_info.add_new_child('vlanid', str(self.parameters['vlanid']))
    vlan_info.add_new_child('node', self.parameters['node'])
    return vlan_info