import numbers
import prettytable
import yaml
from osc_lib import exceptions as exc
from oslo_serialization import jsonutils
def format_dictlist(dict_list):
    string_list = list()
    for mdict in dict_list:
        kv_list = list()
        for k, v in sorted(mdict.items()):
            kv_str = k + ':' + str(v)
            kv_list.append(kv_str)
        this_dict_str = ','.join(kv_list)
        string_list.append(this_dict_str)
    return '\n'.join(string_list)