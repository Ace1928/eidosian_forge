from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_qtree(self):
    """
        Modify a qtree
        """
    if self.use_rest:
        body = self.form_create_modify_body_rest()
        api = 'storage/qtrees/%s' % self.volume_uuid
        query = dict(return_timeout=10)
        dummy, error = rest_generic.patch_async(self.rest_api, api, self.qid, body, query)
        if error:
            self.module.fail_json(msg='Error modifying qtree %s: %s' % (self.parameters['name'], error))
    else:
        self.create_or_modify_qtree_zapi('qtree-modify', 'Error modifying qtree %s: %s')