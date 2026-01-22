from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def initiate_email_callhome(self):
    msg = ''
    email_server_exists = self.check_email_server_exists()
    if email_server_exists:
        self.log('Email server already exists.')
    else:
        self.create_email_server()
        self.changed = True
    self.manage_support_email_user()
    email_user_exists = self.check_email_user_exists()
    if email_user_exists:
        email_user_modify = {}
        if email_user_exists['inventory'] != self.inventory:
            email_user_modify['inventory'] = self.inventory
        if email_user_modify:
            self.update_email_user(email_user_modify, email_user_exists['id'])
    else:
        self.create_email_user()
    self.update_email_data()
    self.enable_email_callhome()
    msg = 'Callhome with email enabled successfully.'
    self.changed = True
    return msg