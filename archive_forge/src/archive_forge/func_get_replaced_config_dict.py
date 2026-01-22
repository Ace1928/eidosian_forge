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
def get_replaced_config_dict(new_conf, exist_conf, test_keys=None, key_set=None):
    replaced_conf = dict()
    if test_keys is None:
        test_keys = []
    if key_set is None:
        key_set = []
    if not new_conf:
        return replaced_conf
    new_key_set = set(new_conf.keys())
    exist_key_set = set(exist_conf.keys())
    trival_new_key_set = set()
    dict_list_new_key_set = set()
    for key in new_key_set:
        if new_conf[key] not in [None, [], {}]:
            if isinstance(new_conf[key], (list, dict)):
                dict_list_new_key_set.add(key)
            else:
                trival_new_key_set.add(key)
    trival_exist_key_set = set()
    dict_list_exist_key_set = set()
    for key in exist_key_set:
        if exist_conf[key] not in [None, [], {}]:
            if isinstance(exist_conf[key], (list, dict)):
                dict_list_exist_key_set.add(key)
            else:
                trival_exist_key_set.add(key)
    common_trival_key_set = trival_new_key_set.intersection(trival_exist_key_set)
    common_dict_list_key_set = dict_list_new_key_set.intersection(dict_list_exist_key_set)
    key_matched_cnt = 0
    common_trival_key_matched = True
    for key in common_trival_key_set:
        if new_conf[key] == exist_conf[key]:
            if key in key_set:
                key_matched_cnt += 1
        elif key not in key_set:
            common_trival_key_matched = False
    for key in common_dict_list_key_set:
        if new_conf[key] == exist_conf[key]:
            if key in key_set:
                key_matched_cnt += 1
    key_matched = key_matched_cnt == len(key_set)
    if key_matched:
        extra_trival_new_key_set = trival_new_key_set - common_trival_key_set
        extra_trival_exist_key_set = trival_exist_key_set - common_trival_key_set
        if extra_trival_new_key_set or extra_trival_exist_key_set or (not common_trival_key_matched):
            replaced_conf = exist_conf
            return replaced_conf
    else:
        replaced_conf = []
        return replaced_conf
    for key in key_set:
        common_dict_list_key_set.discard(key)
    replace_whole_dict = False
    replace_some_list = False
    replace_some_dict = False
    for key in common_dict_list_key_set:
        new_value = new_conf[key]
        exist_value = exist_conf[key]
        if isinstance(new_value, list) and isinstance(exist_value, list):
            n_list = new_value
            e_list = exist_value
            t_keys = next((t_key_item[key] for t_key_item in test_keys if key in t_key_item), None)
            t_key_set = set()
            if t_keys:
                t_key_set = set(t_keys.keys())
            replaced_list = list()
            not_dict_item = False
            dict_no_key_item = False
            for n_item in n_list:
                for e_item in e_list:
                    if isinstance(n_item, dict) and isinstance(e_item, dict):
                        if t_keys:
                            remaining_keys = [t_key_item for t_key_item in test_keys if key not in t_key_item]
                            replaced_dict = get_replaced_config_dict(n_item, e_item, remaining_keys, t_key_set)
                        else:
                            dict_no_key_item = True
                            break
                        if replaced_dict:
                            replaced_list.append(replaced_dict)
                            break
                    else:
                        not_dict_item = True
                        break
                if not_dict_item or dict_no_key_item:
                    break
            if dict_no_key_item:
                replaced_list = e_list
            if not_dict_item:
                n_set = set(n_list)
                e_set = set(e_list)
                diff_set = n_set.symmetric_difference(e_set)
                if diff_set:
                    replaced_conf[key] = e_list
                    replace_some_list = True
            elif replaced_list:
                replaced_conf[key] = replaced_list
                replace_some_list = True
        elif isinstance(new_value, dict) and isinstance(exist_value, dict):
            replaced_dict = get_replaced_config_dict(new_conf[key], exist_conf[key], test_keys)
            if replaced_dict:
                replaced_conf[key] = replaced_dict
                replace_some_dict = True
        elif isinstance(new_value, (list, dict)) or isinstance(exist_value, (list, dict)):
            replaced_conf = exist_conf
            replace_whole_dict = True
            break
        else:
            continue
    if (replace_some_dict or replace_some_list) and (not replace_whole_dict):
        for key in key_set:
            replaced_conf[key] = exist_conf[key]
    return replaced_conf