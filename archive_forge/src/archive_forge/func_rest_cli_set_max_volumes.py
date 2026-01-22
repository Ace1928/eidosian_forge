from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver, zapis_svm
def rest_cli_set_max_volumes(self):
    query = {'vserver': self.parameters['name']}
    body = {'max_volumes': self.parameters['max_volumes']}
    response, error = rest_generic.patch_async(self.rest_api, 'private/cli/vserver', None, body, query)
    if error:
        self.module.fail_json(msg='Error updating max_volumes: %s - %s' % (error, response))