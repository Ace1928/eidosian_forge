from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def call_is_plural(api_call_object, payload):
    is_plural = False
    if 'access' in api_call_object and payload.get('layer') is None:
        is_plural = True
    elif 'threat' in api_call_object and payload.get('layer') is None:
        is_plural = True
    elif 'nat' in api_call_object and payload.get('name') is None and (payload.get('rule-number') is None):
        is_plural = True
    return is_plural