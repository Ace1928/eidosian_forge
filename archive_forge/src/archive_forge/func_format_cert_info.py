from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.openssh.backends.common import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.certificate import (
def format_cert_info(cert_info):
    result = []
    string = ''
    for word in cert_info.split():
        if word in ('Type:', 'Public', 'Signing', 'Key', 'Serial:', 'Valid:', 'Principals:', 'Critical', 'Extensions:'):
            result.append(string)
            string = word
        else:
            string += ' ' + word
    result.append(string)
    result.pop(0)
    return result