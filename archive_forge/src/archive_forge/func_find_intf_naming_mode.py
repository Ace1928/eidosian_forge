from __future__ import absolute_import, division, print_function
import re
import json
import ast
from copy import copy
from itertools import (count, groupby)
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.common.network import (
from ansible.module_utils.common.validation import check_required_arguments
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def find_intf_naming_mode(intf_name):
    ret_intf_naming_mode = NATIVE_MODE
    if re.search(STANDARD_ETH_REGEXP, intf_name):
        ret_intf_naming_mode = STANDARD_MODE
    return ret_intf_naming_mode