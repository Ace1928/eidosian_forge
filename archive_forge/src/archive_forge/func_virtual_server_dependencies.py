from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ipaddress import ip_address
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
@property
def virtual_server_dependencies(self):
    if self.have.virtual_server_dependencies is None:
        return self.want.virtual_server_dependencies
    if self.want.virtual_server_dependencies is None and self.have.virtual_server_dependencies is None:
        return None
    if self.want.virtual_server_dependencies is None:
        return None
    result = compare_complex_list(self.want.virtual_server_dependencies, self.have.virtual_server_dependencies)
    return result