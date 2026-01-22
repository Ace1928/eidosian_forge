from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.argspec.ospf_interfaces.ospf_interfaces import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.rm_templates.ospf_interfaces import (
def get_ospf_interfaces(self, connection, flag):
    cmd = 'show running-config router ' + flag
    return connection.get(cmd)