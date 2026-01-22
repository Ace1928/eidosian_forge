from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def check_if_to_publish_for_action(result, module_args):
    to_publish = ('auto_publish_session' in module_args.keys() and module_args['auto_publish_session']) and ('changed' in result.keys() and result['changed'] is True) and ('failed' not in result.keys() or result['failed'] is False)
    return to_publish