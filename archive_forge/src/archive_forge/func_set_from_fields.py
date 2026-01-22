from __future__ import (absolute_import, division, print_function)
import re
import operator
from functools import reduce
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible.module_utils._text import to_native
def set_from_fields(d, fields, value):
    get_from_fields(d, fields[:-1])[fields[-1]] = value