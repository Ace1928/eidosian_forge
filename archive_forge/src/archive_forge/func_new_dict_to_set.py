from __future__ import absolute_import, division, print_function
import socket
from itertools import count, groupby
from ansible.module_utils.common.network import is_masklen, to_netmask
from ansible.module_utils.six import iteritems
def new_dict_to_set(input_dict, temp_list, test_set, count=0):
    test_dict = dict()
    if isinstance(input_dict, dict):
        input_dict_len = len(input_dict)
        for k, v in sorted(iteritems(input_dict)):
            count += 1
            if isinstance(v, list):
                temp_list.append(k)
                for each in v:
                    if isinstance(each, dict):
                        if [True for i in each.values() if isinstance(i, list)]:
                            new_dict_to_set(each, temp_list, test_set, count)
                        else:
                            new_dict_to_set(each, temp_list, test_set, 0)
            else:
                if v is not None:
                    test_dict.update({k: v})
                try:
                    if tuple(iteritems(test_dict)) not in test_set and count == input_dict_len:
                        test_set.add(tuple(iteritems(test_dict)))
                        count = 0
                except TypeError:
                    temp_dict = {}

                    def expand_dict(dict_to_expand):
                        temp = dict()
                        for k, v in iteritems(dict_to_expand):
                            if isinstance(v, dict):
                                expand_dict(v)
                            else:
                                if v is not None:
                                    temp.update({k: v})
                                temp_dict.update(tuple(iteritems(temp)))
                    new_dict = {k: v}
                    expand_dict(new_dict)
                    if tuple(iteritems(temp_dict)) not in test_set:
                        test_set.add(tuple(iteritems(temp_dict)))
    return test_dict