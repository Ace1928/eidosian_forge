from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def responder_url(self):
    if self.want.responder_url is None:
        return None
    if self.want.responder_url == '' and self.have.responder_url is None:
        return None
    if self.want.responder_url != self.have.responder_url:
        return self.want.responder_url