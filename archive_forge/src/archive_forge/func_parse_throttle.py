from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.ospfv2.ospfv2 import (
def parse_throttle(self, conf, attrib=None):
    """
        This function triggers the parsing of 'throttle' attributes
        :param conf: configuration
        :param attrib: 'spf'
        :return: generated config dictionary
        """
    cfg_dict = {}
    cfg_dict[attrib] = self.parse_attrib(conf, attrib, match=attrib)
    return cfg_dict