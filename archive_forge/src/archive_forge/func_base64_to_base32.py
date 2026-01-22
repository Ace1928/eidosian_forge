from __future__ import absolute_import, division, print_function
import base64
import traceback
from ansible.module_utils.basic import AnsibleModule, sanitize_keys
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def base64_to_base32(base64_string):
    """Converts base64 string to base32 string"""
    b32_string = base64.b32encode(base64.b64decode(base64_string)).decode('ascii')
    return b32_string