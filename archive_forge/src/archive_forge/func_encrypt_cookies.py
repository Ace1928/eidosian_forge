from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def encrypt_cookies(self):
    if self.want.encrypt_cookies is None:
        return None
    if self.have.encrypt_cookies in [None, []]:
        if not self.want.encrypt_cookies:
            return None
        else:
            return self.want.encrypt_cookies
    if set(self.want.encrypt_cookies) != set(self.have.encrypt_cookies):
        return self.want.encrypt_cookies