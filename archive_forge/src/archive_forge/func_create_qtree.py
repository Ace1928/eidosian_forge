from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_qtree(self):
    """
        Create a qtree
        """
    if self.use_rest:
        api = 'storage/qtrees'
        body = {'volume': {'name': self.parameters['flexvol_name']}, 'svm': {'name': self.parameters['vserver']}}
        body.update(self.form_create_modify_body_rest())
        query = dict(return_timeout=10)
        dummy, error = rest_generic.post_async(self.rest_api, api, body, query)
        if error:
            if 'job reported error:' in error and "entry doesn't exist" in error:
                self.module.warn('Ignoring job status, assuming success.')
                return
            self.module.fail_json(msg='Error creating qtree %s: %s' % (self.parameters['name'], error))
    else:
        self.create_or_modify_qtree_zapi('qtree-create', 'Error creating qtree %s: %s')