from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from datetime import datetime
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import (
from ipaddress import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip_interface
from ..module_utils.teem import send_teem
def encode_integer_from_dict(self, record):
    try:
        int(record['key'])
    except ValueError:
        raise F5ModuleError("When specifying an 'integer' type, the value to the left of the separator must be a number.")
    if 'key' in record and 'value' in record:
        return '{0} {1} {2}'.format(record['key'], self._separator, record['value'])
    elif 'key' in record:
        return str(record['key'])