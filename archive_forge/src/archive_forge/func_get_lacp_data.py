from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.lacp.lacp import (
def get_lacp_data(self, connection):
    return connection.get('show lacp sys-id')