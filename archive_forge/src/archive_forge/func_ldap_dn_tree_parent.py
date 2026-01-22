from __future__ import (absolute_import, division, print_function)
import re
def ldap_dn_tree_parent(dn, count=1):
    dn_array = dn.split(',')
    dn_array[0:count] = []
    return ','.join(dn_array)