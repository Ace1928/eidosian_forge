from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def discard_and_fail(self, code, response, connection, version, session_uid):
    discard_code, discard_response = send_request(connection, version, 'discard')
    if discard_code != 200:
        try:
            _fail_json(parse_fail_message(code, response) + ' Failed to discard session {0} with error {1} with message {2}'.format(session_uid, discard_code, discard_response))
        except Exception:
            _fail_json(parse_fail_message(code, response) + ' Failed to discard session with error {0} with message {1}'.format(discard_code, discard_response))
    _fail_json('Checkpoint session with ID: {0}'.format(session_uid) + ', ' + parse_fail_message(code, response) + ' Unpublished changes were discarded')