from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def is_rest_supported_properties(self, parameters, unsupported_rest_properties=None, partially_supported_rest_properties=None, report_error=False):
    used_unsupported_rest_properties = None
    if unsupported_rest_properties:
        used_unsupported_rest_properties = [x for x in unsupported_rest_properties if x in parameters]
    use_rest, error = self.is_rest(used_unsupported_rest_properties, partially_supported_rest_properties, parameters)
    if report_error:
        return (use_rest, error)
    if error:
        self.module.fail_json(msg=error)
    return use_rest