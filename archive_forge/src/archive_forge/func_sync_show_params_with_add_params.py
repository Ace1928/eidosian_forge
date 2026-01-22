from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def sync_show_params_with_add_params(search_result, key_transform):
    temp = {}
    remove_keys = ['type', 'meta-info']
    for k, v in iteritems(search_result):
        if k in remove_keys:
            continue
        if isinstance(v, dict):
            if v.get('name'):
                temp.update({k: v['name']})
            else:
                temp_child = {}
                for each_k, each_v in iteritems(v):
                    if isinstance(each_v, dict):
                        if each_v.get('name'):
                            temp_child.update({each_k: each_v['name']})
                    else:
                        temp_child.update({each_k: each_v})
                temp.update({k: temp_child})
        elif isinstance(v, list):
            temp[k] = []
            for each in v:
                if each.get('name'):
                    temp[k].append(each['name'])
                else:
                    temp.update(each)
        else:
            temp.update({k: v})
    temp = map_obj_to_params(temp, key_transform, '')
    return temp