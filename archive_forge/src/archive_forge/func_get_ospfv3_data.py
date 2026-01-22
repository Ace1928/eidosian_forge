from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.ospfv3.ospfv3 import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.ospfv3 import (
def get_ospfv3_data(self, connection):
    return connection.get('show running-config | section ^router ospfv3')