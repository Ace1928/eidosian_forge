from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def key_filename(self):
    true_name = flatten_boolean(self.true_names)
    if true_name == 'yes':
        return self.name
    elif self.name.endswith('.key'):
        return self.name
    else:
        return self.name + '.key'