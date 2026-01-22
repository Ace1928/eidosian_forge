from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def update_email_data(self):
    if self.module.check_mode:
        self.changed = True
        return
    command = 'chemail'
    command_options = {}
    if self.contact_email:
        command_options['reply'] = self.contact_email
    if self.contact_name:
        command_options['contact'] = self.contact_name
    if self.phonenumber_primary:
        command_options['primary'] = self.phonenumber_primary
    if self.phonenumber_secondary:
        command_options['alternate'] = self.phonenumber_secondary
    if self.location:
        command_options['location'] = self.location
    if self.company_name:
        command_options['organization'] = self.company_name
    if self.address:
        command_options['address'] = self.address
    if self.city:
        command_options['city'] = self.city
    if self.province:
        command_options['state'] = self.province
    if self.postalcode:
        command_options['zip'] = self.postalcode
    if self.country:
        command_options['country'] = self.country
    cmdargs = None
    if command_options:
        self.restapi.svc_run_command(command, command_options, cmdargs)
        self.log('Email data successfully updated.')