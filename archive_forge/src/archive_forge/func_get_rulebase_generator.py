from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def get_rulebase_generator(connection, version, show_rulebase_identifier_payload, show_rulebase_command, rules_amount):
    offset = 0
    limit = 100
    while True:
        payload_for_show_rulebase = {'limit': limit, 'offset': offset}
        payload_for_show_rulebase.update(show_rulebase_identifier_payload)
        if offset + limit >= rules_amount:
            del payload_for_show_rulebase['limit']
        code, response = send_request(connection, version, show_rulebase_command, payload_for_show_rulebase)
        offset = response['to']
        total = response['total']
        rulebase = response['rulebase']
        yield rulebase
        if total <= offset:
            return