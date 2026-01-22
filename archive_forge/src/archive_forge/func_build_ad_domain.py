from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def build_ad_domain(self):
    ad_domain = {}
    if 'admin_user_name' in self.parameters:
        ad_domain['user'] = self.parameters['admin_user_name']
    if 'admin_password' in self.parameters:
        ad_domain['password'] = self.parameters['admin_password']
    if 'ou' in self.parameters:
        ad_domain['organizational_unit'] = self.parameters['ou']
    if 'domain' in self.parameters:
        ad_domain['fqdn'] = self.parameters['domain']
    if 'default_site' in self.parameters:
        ad_domain['default_site'] = self.parameters['default_site']
    return ad_domain