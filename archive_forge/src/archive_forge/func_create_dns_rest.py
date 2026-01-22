from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def create_dns_rest(self):
    """
        Create DNS server
        :return: none
        """
    if self.is_cluster or not self.parameters.get('vserver'):
        return self.patch_cluster_dns()
    api = 'name-services/dns'
    body = {'domains': self.parameters['domains'], 'servers': self.parameters['nameservers'], 'svm': {'name': self.parameters['vserver']}}
    if 'skip_validation' in self.parameters:
        body['skip_config_validation'] = self.parameters['skip_validation']
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating DNS service: %s' % error)