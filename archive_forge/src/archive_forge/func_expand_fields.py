from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def expand_fields(fields, item, class_name):
    class_ = getattr(spotinst.aws_elastigroup, class_name)
    new_obj = class_()
    if item is not None:
        for field in fields:
            if isinstance(field, dict):
                ansible_field_name = field['ansible_field_name']
                spotinst_field_name = field['spotinst_field_name']
            else:
                ansible_field_name = field
                spotinst_field_name = field
            if item.get(ansible_field_name) is not None:
                setattr(new_obj, spotinst_field_name, item.get(ansible_field_name))
    return new_obj