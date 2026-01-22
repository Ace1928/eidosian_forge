from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def express_server(self):
    if self.want.express_server is None:
        return None
    if self.want.express_server == '' and self.have.express_server is None:
        return None
    if self.want.express_server != self.have.express_server:
        return self.want.express_server