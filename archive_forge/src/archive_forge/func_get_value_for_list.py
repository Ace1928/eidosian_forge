from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import re
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def get_value_for_list(self, from_zapi, zapi_parent, zapi_child=None, data=None):
    """
        Convert a python list() to NaElement or vice-versa
        If from_zapi = True, value is converted from NaElement (parent-children structure) to list()
        If from_zapi = False, value is converted from list() to NaElement
        :param zapi_parent: ZAPI parent key or the ZAPI parent NaElement
        :param zapi_child: ZAPI child key
        :param data: list() to be converted to NaElement parent-children object
        :param from_zapi: convert the value from ZAPI or to ZAPI acceptable type
        :return: list() or NaElement
        """
    if from_zapi:
        if zapi_parent is None:
            return []
        return [zapi_child.get_content() for zapi_child in zapi_parent.get_children()]
    zapi_parent = netapp_utils.zapi.NaElement(zapi_parent)
    for item in data:
        zapi_parent.add_new_child(zapi_child, item)
    return zapi_parent