from __future__ import absolute_import, division, print_function
import codecs
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_bytes, to_native, to_text
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def convert_keys(d_param):
    """Method to convert hyphen to underscore"""
    if isinstance(d_param, dict):
        out = {}
        for key, val in d_param.items():
            val = convert_keys(val)
            out[key.replace('-', '_')] = val
        return out
    elif isinstance(d_param, list):
        return [convert_keys(val) for val in d_param]
    return d_param