from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def is_ldapgroup_exists(self, uid):
    return self.ldap_interface.exists(uid)