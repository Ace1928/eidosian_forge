from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.lldp_interfaces.lldp_interfaces import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.utils.utils import (
def get_lldp_interfaces_data(self, connection):
    return connection.get('show lldp interface')