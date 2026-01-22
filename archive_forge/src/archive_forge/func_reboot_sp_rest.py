from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def reboot_sp_rest(self):
    uuid = self.get_node_uuid()
    api = 'cluster/nodes'
    body = {'service_processor.action': 'reboot'}
    dummy, error = rest_generic.patch_async(self.rest_api, api, uuid, body)
    if error and 'Unexpected argument "service_processor.action"' in error:
        error = self.reboot_sp_rest_cli()
        if error:
            error = 'reboot_sp requires ONTAP 9.10.1 or newer, falling back to CLI passthrough failed: ' + error
    if error:
        self.module.fail_json(msg='Error rebooting node SP: %s' % error)