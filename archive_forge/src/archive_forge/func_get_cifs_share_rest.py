from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_cifs_share_rest(self):
    """
        get uuid of the svm which has CIFS share with rest API.
        """
    options = {'svm.name': self.parameters.get('vserver'), 'name': self.parameters.get('share_name')}
    api = 'protocols/cifs/shares'
    fields = 'svm.uuid,name'
    record, error = rest_generic.get_one_record(self.rest_api, api, options, fields)
    if error:
        self.module.fail_json(msg='Error on fetching cifs shares: %s' % error)
    if record:
        return {'uuid': record['svm']['uuid']}
    self.module.fail_json(msg='Error: the cifs share does not exist: %s' % self.parameters['share_name'])