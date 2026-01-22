from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def wait_on_work_request(client, response, module):
    try:
        if module.params.get('wait', None):
            _debug('Waiting for work request with id {0} to reach SUCCEEDED state.'.format(response.data.id))
            wait_response = oci.wait_until(client, response, evaluate_response=lambda r: r.data.status == 'SUCCEEDED', max_wait_seconds=module.params.get('wait_timeout', MAX_WAIT_TIMEOUT_IN_SECONDS))
        else:
            _debug('Waiting for work request with id {0} to reach ACCEPTED state.'.format(response.data.id))
            wait_response = oci.wait_until(client, response, evaluate_response=lambda r: r.data.status == 'ACCEPTED', max_wait_seconds=module.params.get('wait_timeout', MAX_WAIT_TIMEOUT_IN_SECONDS))
    except MaximumWaitTimeExceeded as ex:
        _debug(str(ex))
        module.fail_json(msg=str(ex))
    except ServiceError as ex:
        _debug(str(ex))
        module.fail_json(msg=str(ex))
    return wait_response.data