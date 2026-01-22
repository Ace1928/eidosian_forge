from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def build_payload(api_call_object, payload, params_to_remove):
    if api_call_object in params_to_remove:
        for param in params_to_remove[api_call_object]:
            del payload[param]
    return payload