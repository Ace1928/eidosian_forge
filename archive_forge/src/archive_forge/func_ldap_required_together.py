from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.common.text.converters import to_native
def ldap_required_together():
    return [['client_cert', 'client_key']]