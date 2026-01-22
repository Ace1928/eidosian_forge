from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_unix_group_rest(self):
    """
        Retrieves the UNIX groups for all of the SVMs.
        UNIX users who are the members of the group are also displayed.
        """
    if not self.use_rest:
        return self.get_unix_group()
    query = {'svm.name': self.parameters.get('vserver'), 'name': self.parameters.get('name')}
    api = 'name-services/unix-groups'
    fields = 'svm.uuid,id,name,users.name'
    record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg='Error getting UNIX group: %s' % error)
    if record:
        if 'users' in record:
            record['users'] = [user['name'] for user in record['users']]
        return {'svm': {'uuid': self.na_helper.safe_get(record, ['svm', 'uuid'])}, 'name': self.na_helper.safe_get(record, ['name']), 'id': self.na_helper.safe_get(record, ['id']), 'users': self.na_helper.safe_get(record, ['users'])}
    return None