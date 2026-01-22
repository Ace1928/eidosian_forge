from __future__ import absolute_import, division, print_function
import codecs
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text, to_bytes
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_owning_resource, rest_vserver
def private_cli_fields(self, api):
    """
        The private cli endpoint does not allow '*' to be an entered.
        If fields='*' or fields are not included within the playbook, the API call will be populated to return all possible fields.
        If fields is entered into the playbook the fields entered will be used when calling the API.
        """
    if 'fields' not in self.parameters or '*' in self.parameters['fields'] or '**' in self.parameters['fields']:
        if api == 'support/autosupport/check':
            fields = 'node,corrective-action,status,error-detail,check-type,check-category'
        elif api == 'private/cli/vserver/security/file-directory':
            fields = 'acls'
        else:
            self.module.fail_json(msg='Internal error, no field for %s' % api)
    else:
        fields = ','.join(self.parameters['fields'])
    return fields