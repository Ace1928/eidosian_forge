from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import ip_network
from ..module_utils.teem import send_teem
def check_required_params(self):
    if self.want.split_tunnel == 'true':
        if self.want.ip_version == 'ipv4':
            if self.want.ipv4_address_space in [None, []]:
                raise F5ModuleError('The ipv4_address_space cannot be empty, when split_tunnel is set to {0}'.format(self.want.split_tunnel))
        if self.want.ip_version == 'ipv4-ipv6':
            if self.want.ipv4_address_space in [None, []]:
                raise F5ModuleError('The ipv4_address_space cannot be empty, when split_tunnel is set to {0}'.format(self.want.split_tunnel))
            if self.want.ipv6_address_space in [None, []]:
                raise F5ModuleError('The ipv6_address_space cannot be empty, when split_tunnel is set to {0}'.format(self.want.split_tunnel))
    if self.have.split_tunnel == 'true':
        if self.have.ip_version == 'ipv4':
            if self.want.ipv4_address_space is not None and (not self.want.ipv4_address_space):
                raise F5ModuleError('Cannot remove ipv4_address_space when split_tunnel on device is: {0}'.format(self.have.split_tunnel))
        if self.have.ip_version == 'ipv4-ipv6':
            if self.want.ipv4_address_space is not None and (not self.want.ipv4_address_space):
                raise F5ModuleError('Cannot remove ipv4_address_space when split_tunnel on device is: {0}'.format(self.have.split_tunnel))
            if self.want.ipv6_address_space is not None and (not self.want.ipv6_address_space):
                raise F5ModuleError('Cannot remove ipv6_address_space when split_tunnel on device is: {0}'.format(self.have.split_tunnel))