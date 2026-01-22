from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def is_valid_as_number(as_number):
    """check as-number is valid"""
    if as_number.isdigit():
        if int(as_number) > 4294967295 or int(as_number) < 1:
            return False
        return True
    else:
        if as_number.find('.') != -1:
            number_list = as_number.split('.')
            if len(number_list) != 2:
                return False
            if number_list[1] == 0:
                return False
            for each_num in number_list:
                if not each_num.isdigit():
                    return False
                if int(each_num) > 65535:
                    return False
            return True
        return False