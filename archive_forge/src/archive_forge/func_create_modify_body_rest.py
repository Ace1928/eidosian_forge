from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_modify_body_rest(self, params=None):
    """
        Function to define body for create and modify cifs server
        """
    body, query, security, service_options = ({}, {}, {}, {})
    if params is None:
        params = self.parameters
    security_options = ['smb_signing', 'encrypt_dc_connection', 'kdc_encryption', 'smb_encryption', 'restrict_anonymous', 'aes_netlogon_enabled', 'ldap_referral_enabled', 'try_ldap_channel_binding', 'session_security', 'lm_compatibility_level', 'use_ldaps', 'use_start_tls']
    ad_domain = self.build_ad_domain()
    if ad_domain:
        body['ad_domain'] = ad_domain
    if 'force' in self.parameters:
        query['force'] = self.parameters['force']
    for key in security_options:
        if key in params:
            security[key] = params[key]
    if security:
        body['security'] = security
    for key, option in [('multichannel', 'is_multichannel_enabled')]:
        if option in params:
            service_options.update({key: params[option]})
    if service_options:
        body['options'] = service_options
    if 'vserver' in params:
        body['svm.name'] = params['vserver']
    if 'cifs_server_name' in params:
        body['name'] = self.parameters['cifs_server_name']
    if 'service_state' in params:
        body['enabled'] = params['service_state'] == 'started'
    return (body, query)