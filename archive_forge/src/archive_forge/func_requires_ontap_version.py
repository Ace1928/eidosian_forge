from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def requires_ontap_version(self, module_name, version='9.6'):
    suffix = ' - %s' % self.is_rest_error if self.is_rest_error is not None else ''
    return '%s only supports REST, and requires ONTAP %s or later.%s' % (module_name, version, suffix)