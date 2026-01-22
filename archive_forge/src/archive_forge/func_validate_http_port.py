from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.arista.eos.plugins.module_utils.network.eos.eos import (
def validate_http_port(value, module):
    if not 1 <= value <= 65535:
        module.fail_json(msg='http_port must be between 1 and 65535')