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
def encode_string_from_string(self, record):
    parts = record.split(self._separator)
    if len(parts) == 2:
        return '{0} {1} {2}'.format(parts[0], self._separator, parts[1])
    elif len(parts) == 1 and parts[0] != '':
        return '{0} {1} ""'.format(parts[0], self._separator)