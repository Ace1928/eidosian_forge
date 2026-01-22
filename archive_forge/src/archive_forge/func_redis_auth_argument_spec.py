from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
import traceback
def redis_auth_argument_spec(tls_default=True):
    return dict(login_host=dict(type='str', default='localhost'), login_user=dict(type='str'), login_password=dict(type='str', no_log=True), login_port=dict(type='int', default=6379), tls=dict(type='bool', default=tls_default), validate_certs=dict(type='bool', default=True), ca_certs=dict(type='str'))