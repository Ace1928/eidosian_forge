from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def openshift_equal_dn_objects(dn_obj, other_dn_obj):
    if len(dn_obj) != len(other_dn_obj):
        return False
    for k, v in enumerate(dn_obj):
        if len(v) != len(other_dn_obj[k]):
            return False
        for j, item in enumerate(v):
            if not item == other_dn_obj[k][j]:
                return False
    return True