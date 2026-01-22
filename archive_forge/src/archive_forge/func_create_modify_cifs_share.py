from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_modify_cifs_share(self, zapi_request, action):
    if self.parameters.get('share_properties'):
        property_attrs = netapp_utils.zapi.NaElement('share-properties')
        zapi_request.add_child_elem(property_attrs)
        for aproperty in self.parameters.get('share_properties'):
            property_attrs.add_new_child('cifs-share-properties', aproperty)
    if self.parameters.get('symlink_properties'):
        symlink_attrs = netapp_utils.zapi.NaElement('symlink-properties')
        zapi_request.add_child_elem(symlink_attrs)
        for symlink in self.parameters.get('symlink_properties'):
            symlink_attrs.add_new_child('cifs-share-symlink-properties', symlink)
    if self.parameters.get('vscan_fileop_profile'):
        fileop_attrs = netapp_utils.zapi.NaElement('vscan-fileop-profile')
        fileop_attrs.set_content(self.parameters['vscan_fileop_profile'])
        zapi_request.add_child_elem(fileop_attrs)
    if self.parameters.get('comment'):
        zapi_request.add_new_child('comment', self.parameters['comment'])
    try:
        self.server.invoke_successfully(zapi_request, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error %s cifs-share %s: %s' % (action, self.parameters.get('name'), to_native(error)), exception=traceback.format_exc())