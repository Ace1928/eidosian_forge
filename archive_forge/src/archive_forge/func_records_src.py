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
def records_src(self):
    try:
        self._values['records_src'].seek(0)
        return self._values['records_src']
    except AttributeError:
        pass
    if self._values['records_src']:
        records = open(self._values['records_src'])
    else:
        records = self._values['records']
    if records is None:
        return None
    self._values['records_src'] = StringIO()
    self._write_records_to_file(records)
    return self._values['records_src']