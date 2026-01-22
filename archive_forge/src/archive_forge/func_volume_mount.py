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
def volume_mount(self):
    """
        Mount an existing volume in specified junction_path
        :return: None
        """
    if self.use_rest:
        return self.volume_mount_rest()
    vol_mount = netapp_utils.zapi.NaElement('volume-mount')
    vol_mount.add_new_child('volume-name', self.parameters['name'])
    vol_mount.add_new_child('junction-path', self.parameters['junction_path'])
    try:
        self.server.invoke_successfully(vol_mount, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error mounting volume %s on path %s: %s' % (self.parameters['name'], self.parameters['junction_path'], to_native(error)), exception=traceback.format_exc())