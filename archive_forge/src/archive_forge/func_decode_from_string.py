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
def decode_from_string(self, record):
    parts = record.split(self._separator)
    if len(parts) == 2:
        return dict(name=parts[0].strip(), data=parts[1].strip().strip('"'))
    else:
        return dict(name=parts[0].strip(), data='')