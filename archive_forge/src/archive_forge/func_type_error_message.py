from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import re
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
@staticmethod
def type_error_message(type_str, key, value):
    return "expecting '%s' type for %s: %s, got: %s" % (type_str, repr(key), repr(value), type(value))