from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def contains_show_identifier_param(payload):
    identifier_params = ['name', 'uid', 'assigned-domain', 'task-id', 'signature', 'url']
    for param in identifier_params:
        if payload.get(param) is not None:
            return True
    return False