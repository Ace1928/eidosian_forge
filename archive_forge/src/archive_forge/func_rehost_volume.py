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
def rehost_volume(self):
    volume_rehost = netapp_utils.zapi.NaElement.create_node_with_children('volume-rehost', **{'vserver': self.parameters['from_vserver'], 'destination-vserver': self.parameters['vserver'], 'volume': self.parameters['name']})
    if self.parameters.get('auto_remap_luns') is not None:
        volume_rehost.add_new_child('auto-remap-luns', str(self.parameters['auto_remap_luns']))
    if self.parameters.get('force_unmap_luns') is not None:
        volume_rehost.add_new_child('force-unmap-luns', str(self.parameters['force_unmap_luns']))
    try:
        self.cluster.invoke_successfully(volume_rehost, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error rehosting volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())