from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_aggr_object_store(self, body):
    """
        Create aggregate object store config
        :return: None
        """
    if self.use_rest:
        api = 'cloud/targets'
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
        if error:
            self.module.fail_json(msg='Error %s' % error)
    else:
        options = {'object-store-name': self.parameters['name'], 'provider-type': self.parameters['provider_type'], 'server': self.parameters['server'], 's3-name': self.parameters['container'], 'access-key': self.parameters['access_key']}
        if self.parameters.get('secret_password'):
            options['secret-password'] = self.parameters['secret_password']
        if self.parameters.get('port') is not None:
            options['port'] = str(self.parameters['port'])
        if self.parameters.get('certificate_validation_enabled') is not None:
            options['is-certificate-validation-enabled'] = str(self.parameters['certificate_validation_enabled']).lower()
        if self.parameters.get('ssl_enabled') is not None:
            options['ssl-enabled'] = str(self.parameters['ssl_enabled']).lower()
        object_store_create = netapp_utils.zapi.NaElement.create_node_with_children('aggr-object-store-config-create', **options)
        try:
            self.server.invoke_successfully(object_store_create, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error provisioning object store config %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())