from __future__ import absolute_import, division, print_function
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.ecs.api import (
def ecs_domain_argument_spec():
    return dict(client_id=dict(type='int', default=1), domain_name=dict(type='str', required=True), verification_method=dict(type='str', required=True, choices=['dns', 'email', 'manual', 'web_server']), verification_email=dict(type='str'))