from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def route_domain(self):
    if self.want.route_domain is None:
        return None
    if self.have.route_domain != self.want.route_domain:
        if self.want.route_domain == 0 and self.want.ipv4_interface:
            return dict(tunnel_local_address='any', tunnel_remote_address='any', route_domain=self.want.route_domain)
        elif self.want.route_domain == 0 and (not self.want.ipv4_interface):
            return dict(tunnel_local_address='any6', tunnel_remote_address='any6', route_domain=self.want.route_domain)
        else:
            return dict(tunnel_local_address='any%{0}'.format(self.want.route_domain), tunnel_remote_address='any%{0}'.format(self.want.route_domain), route_domain=self.want.route_domain)