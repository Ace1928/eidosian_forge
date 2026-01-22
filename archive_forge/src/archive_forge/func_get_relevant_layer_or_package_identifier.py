from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def get_relevant_layer_or_package_identifier(api_call_object, payload):
    if 'nat' in api_call_object:
        identifier = {'package': payload['package']}
    else:
        identifier = {'layer': payload['layer']}
    return identifier