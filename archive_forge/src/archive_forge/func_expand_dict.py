from __future__ import absolute_import, division, print_function
import socket
from itertools import count, groupby
from ansible.module_utils.common.network import is_masklen, to_netmask
from ansible.module_utils.six import iteritems
def expand_dict(dict_to_expand):
    temp = dict()
    for k, v in iteritems(dict_to_expand):
        if isinstance(v, dict):
            expand_dict(v)
        else:
            if v is not None:
                temp.update({k: v})
            temp_dict.update(tuple(iteritems(temp)))