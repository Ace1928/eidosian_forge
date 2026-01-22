from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def create_active_directory_preferred_domain_controllers_rest(self):
    """
        Adds the Active Directory preferred DC configuration to an SVM.
        """
    query = {}
    if self.rest_api.meets_rest_minimum_version(True, 9, 12, 0):
        api = 'protocols/active-directory/%s/preferred-domain-controllers' % self.svm_uuid
        body = {'fqdn': self.parameters['fqdn'], 'server_ip': self.parameters['server_ip']}
        if 'skip_config_validation' in self.parameters:
            query['skip_config_validation'] = self.parameters['skip_config_validation']
    else:
        api = 'private/cli/vserver/active-directory/preferred-dc/add'
        body = {'vserver': self.parameters['vserver'], 'domain': self.parameters['fqdn'], 'preferred_dc': [self.parameters['server_ip']]}
        if 'skip_config_validation' in self.parameters:
            query['skip_config_validation'] = self.parameters['skip_config_validation']
    dummy, error = rest_generic.post_async(self.rest_api, api, body, query)
    if error:
        self.module.fail_json(msg='Error on adding Active Directory preferred DC configuration to an SVM: %s' % error)