from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.ospfv2.ospfv2 import (
def parse_ospf(self, conf, attrib=None):
    """
        This function triggers the parsing of 'distance ospf' attributes
        :param conf: configuration
        :param attrib: 'ospf'
        :return: generated config dictionary
        """
    cfg_dict = self.parse_attrib(conf, 'ospf', match=attrib)
    return cfg_dict