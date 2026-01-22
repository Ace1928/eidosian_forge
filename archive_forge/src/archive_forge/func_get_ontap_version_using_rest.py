from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def get_ontap_version_using_rest(self):
    method = 'GET'
    api = 'cluster'
    params = {'fields': ['version']}
    status_code, message, error = self.send_request(method, api, params=params)
    try:
        if error and 'are available in precluster.' in error.get('message', ''):
            status_code, message, error = self.get_node_version_using_rest()
    except AttributeError:
        pass
    self.set_version(message)
    if error:
        self.log_error(status_code, str(error))
    if self.force_ontap_version:
        warning = self.get_ontap_version_from_params()
        if error:
            warning += ' error: %s, status_code: %s' % (error, status_code)
        if warning:
            self.module.warn(warning)
            msg = 'Forcing ONTAP version to %s' % self.force_ontap_version
            if error:
                self.log_error('INFO', msg)
            else:
                self.log_debug('INFO', msg)
        error = None
        status_code = 200
    self.is_rest_error = str(error) if error else None
    return status_code