from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.ospfv2.ospfv2 import (
def map_regex(self, attrib):
    """
        - This function construct the regex string.
        - replace the underscore with hyphen.
        :param attrib: attribute
        :return: regex string
        """
    return 'disable' if attrib == 'disabled' else 'enable' if attrib == 'enabled' else 'area' if attrib == 'area_id' else attrib.replace('_', '-')