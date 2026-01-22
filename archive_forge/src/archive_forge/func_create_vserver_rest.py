from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver, zapis_svm
def create_vserver_rest(self):
    body, allowed_protocols = self.create_body_contents()
    dummy, error = rest_generic.post_async(self.rest_api, 'svm/svms', body, timeout=self.timeout)
    if error:
        self.module.fail_json(msg='Error in create: %s' % error)
    if self.parameters.get('max_volumes') is not None and (not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1)):
        self.rest_cli_set_max_volumes()
    if allowed_protocols:
        self.rest_cli_add_remove_protocols(allowed_protocols)