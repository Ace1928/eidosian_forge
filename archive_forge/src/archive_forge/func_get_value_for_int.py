from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import re
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def get_value_for_int(self, from_zapi, value, key=None):
    """
        Convert integer values to string or vice-versa
        If from_zapi = True, value is converted from string (as it appears in ZAPI) to integer
        If from_zapi = False, value is converted from integer to string
        For get() method, from_zapi = True
        For modify(), create(), from_zapi = False
        :param from_zapi: convert the value from ZAPI or to ZAPI acceptable type
        :param value: value of the integer attribute
        :param key: if present, force error checking to validate type
        :return: string or integer
        """
    if value is None:
        return None
    if from_zapi:
        if key is not None and (not isinstance(value, str)):
            raise TypeError(self.type_error_message('str', key, value))
        return int(value)
    if key is not None and (not isinstance(value, int)):
        raise TypeError(self.type_error_message('int', key, value))
    return str(value)