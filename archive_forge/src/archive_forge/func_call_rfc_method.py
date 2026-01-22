from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
import traceback
import datetime
def call_rfc_method(connection, method_name, kwargs):
    return connection.call(method_name, **kwargs)