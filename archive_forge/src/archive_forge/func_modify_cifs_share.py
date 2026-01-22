from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_cifs_share(self):
    """
        modilfy path for the given CIFS share
        """
    options = {'share-name': self.parameters.get('name')}
    cifs_modify = netapp_utils.zapi.NaElement.create_node_with_children('cifs-share-modify', **options)
    if self.parameters.get('path'):
        cifs_modify.add_new_child('path', self.parameters.get('path'))
    self.create_modify_cifs_share(cifs_modify, 'modifying')