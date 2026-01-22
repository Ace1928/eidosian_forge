from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def na_ontap_host_argument_spec_peer():
    spec = na_ontap_host_argument_spec()
    spec.pop('feature_flags')
    for value in spec.values():
        if 'default' in value:
            value.pop('default')
    return spec