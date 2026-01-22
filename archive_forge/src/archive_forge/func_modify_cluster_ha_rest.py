from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_cluster_ha_rest(self, configure):
    api = 'private/cli/cluster/ha'
    body = {'configured': True if configure == 'true' else False}
    dummy, error = rest_generic.patch_async(self.rest_api, api, None, body)
    if error:
        self.module.fail_json(msg='Error modifying cluster HA to %s: %s' % (configure, to_native(error)))