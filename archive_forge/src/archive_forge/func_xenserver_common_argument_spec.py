from __future__ import absolute_import, division, print_function
import atexit
import time
import re
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.ansible_release import __version__ as ANSIBLE_VERSION
def xenserver_common_argument_spec():
    return dict(hostname=dict(type='str', aliases=['host', 'pool'], required=False, default='localhost', fallback=(env_fallback, ['XENSERVER_HOST'])), username=dict(type='str', aliases=['user', 'admin'], required=False, default='root', fallback=(env_fallback, ['XENSERVER_USER'])), password=dict(type='str', aliases=['pass', 'pwd'], required=False, no_log=True, fallback=(env_fallback, ['XENSERVER_PASSWORD'])), validate_certs=dict(type='bool', required=False, default=True, fallback=(env_fallback, ['XENSERVER_VALIDATE_CERTS'])))