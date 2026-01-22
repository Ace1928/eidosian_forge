from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def get_existing_email_user_data(self):
    data = {}
    email_data = self.restapi.svc_obj_info(cmd='lsemailuser', cmdopts=None, cmdargs=None)
    for item in email_data:
        if item['address'] == self.contact_email:
            data = item
    return data