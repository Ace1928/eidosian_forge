from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_certificate_serial(self, cert_name):
    """Retrieve the serial of a certificate"""
    api = 'security/certificates'
    query = {'scope': 'cluster', 'type': 'client', 'name': cert_name}
    fields = 'serial_number'
    record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg='Error retrieving certificates: %s' % error)
    if not record:
        self.module.fail_json(msg='Error certificate not found: %s.' % self.parameters['certificate'])
    return record['serial_number']