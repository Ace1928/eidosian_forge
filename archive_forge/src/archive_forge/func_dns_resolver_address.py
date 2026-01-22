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
def dns_resolver_address(self):
    resolver = self.dns_resolver
    if resolver is None:
        return None
    tmp = resolver.split('/')
    link = dict(link='https://localhost/mgmt/tm/net/dns-resolver/~{0}~{1}'.format(tmp[1], tmp[2]))
    return link