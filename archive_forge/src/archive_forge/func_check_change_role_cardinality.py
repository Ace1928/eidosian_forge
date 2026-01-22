from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
def check_change_role_cardinality(module, service, role_name, cardinality):
    roles_list = service['TEMPLATE']['BODY']['roles']
    for role in roles_list:
        if role['name'] == role_name:
            return int(role['cardinality']) != cardinality
    module.fail_json(msg='There is no role with name: ' + role_name)