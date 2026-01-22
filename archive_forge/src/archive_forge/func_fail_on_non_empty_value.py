from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def fail_on_non_empty_value(response):
    """json() may fail on an empty value, but it's OK if no response is expected.
               To avoid false positives, only report an issue when we expect to read a value.
               The first get will see it.
            """
    if method == 'GET' and has_feature(self.module, 'strict_json_check'):
        contents = response.content
        if len(contents) > 0:
            raise ValueError('Expecting json, got: %s' % contents)