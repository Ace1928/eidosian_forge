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
def get_ranges_in_list(num_list):
    """Returns a generator for list(s) of consecutive numbers
    present in the given sorted list of numbers
    """
    for key, group in groupby(num_list, lambda num, i=count(): num - next(i)):
        yield list(group)