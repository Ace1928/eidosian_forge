from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.ospfv2.ospfv2 import (
def parse_auto_cost(self, conf, attrib=None):
    """
        This function triggers the parsing of 'auto_cost' attributes
        :param conf: configuration
        :param attrib: attribute name
        :return: generated config dictionary
        """
    cfg_dict = self.parse_attr(conf, ['reference_bandwidth'], match=attrib)
    return cfg_dict