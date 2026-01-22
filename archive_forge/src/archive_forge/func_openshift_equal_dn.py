from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def openshift_equal_dn(dn, other):
    dn_obj = ldap.dn.str2dn(dn)
    other_dn_obj = ldap.dn.str2dn(other)
    return openshift_equal_dn_objects(dn_obj, other_dn_obj)