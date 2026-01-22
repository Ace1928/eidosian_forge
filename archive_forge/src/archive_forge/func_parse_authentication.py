from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.ospfv2.ospfv2 import (
def parse_authentication(self, conf, attrib=None):
    """
        This function triggers the parsing of 'authentication' attributes.
        :param conf: configuration
        :param attrib: 'authentication'
        :return: generated config dictionary
        """
    cfg_dict = self.parse_attr(conf, ['plaintext_password'], match=attrib)
    cfg_dict['md5'] = self.parse_attrib_list(conf, 'key-id', 'key_id')
    return cfg_dict