from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_ems_destination(self):
    api = 'support/ems/destinations'
    name = self.parameters['name']
    body = {'name': name, 'type': self.parameters['type'], 'destination': self.parameters['destination'], 'filters': self.generate_filters_list(self.parameters['filters'])}
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 11, 1):
        if self.parameters.get('certificate') and self.parameters.get('ca') is not None:
            body['certificate'] = {'serial_number': self.get_certificate_serial(self.parameters['certificate']), 'ca': self.parameters['ca']}
    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 12, 1):
        if self.parameters.get('syslog') is not None:
            body['syslog'] = {}
            for key, option in [('syslog.port', 'port'), ('syslog.transport', 'transport'), ('syslog.format.message', 'message_format'), ('syslog.format.timestamp_override', 'timestamp_format_override'), ('syslog.format.hostname_override', 'hostname_format_override')]:
                if self.parameters['syslog'].get(option) is not None:
                    body[key] = self.parameters['syslog'][option]
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    self.fail_on_error(error, 'creating EMS destinations for %s' % name)