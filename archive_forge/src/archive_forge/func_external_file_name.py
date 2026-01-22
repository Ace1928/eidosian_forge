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
@property
def external_file_name(self):
    if self._values['external_file_name'] is None:
        name = self.name
    else:
        name = self._values['external_file_name']
    if re.search('[^a-zA-Z0-9-_.]', name):
        raise F5ModuleError("'external_file_name' may only contain letters, numbers, underscores, dashes, or a period.")
    return name