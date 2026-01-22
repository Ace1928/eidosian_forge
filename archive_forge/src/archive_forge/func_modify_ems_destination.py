from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_ems_destination(self, name, modify):
    if 'type' in modify:
        self.delete_ems_destination(name)
        self.create_ems_destination()
    else:
        body = {}
        if any((item in modify for item in ['certificate', 'ca'])):
            body['certificate'] = {}
        for option in modify:
            if option == 'filters':
                body[option] = self.generate_filters_list(modify[option])
            elif option == 'certificate':
                body[option]['serial_number'] = self.get_certificate_serial(modify[option])
            elif option == 'ca':
                body['certificate']['ca'] = modify[option]
            elif option == 'syslog':
                for key, option in [('syslog.port', 'port'), ('syslog.transport', 'transport'), ('syslog.format.message', 'message_format'), ('syslog.format.timestamp_override', 'timestamp_format_override'), ('syslog.format.hostname_override', 'hostname_format_override')]:
                    if option in modify['syslog']:
                        body[key] = modify['syslog'][option]
            else:
                body[option] = modify[option]
        if body:
            api = 'support/ems/destinations'
            dummy, error = rest_generic.patch_async(self.rest_api, api, name, body)
            self.fail_on_error(error, 'modifying EMS destination for %s' % name)