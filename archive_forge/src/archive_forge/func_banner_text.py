from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def banner_text(self):
    if self.want.banner_text is None:
        return None
    if self.want.banner_text == '' and self.have.banner_text is None:
        return None
    if self.want.banner_text != self.have.banner_text:
        return self.want.banner_text