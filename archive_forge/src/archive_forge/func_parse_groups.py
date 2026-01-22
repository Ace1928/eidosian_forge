from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_global.firewall_global import (
def parse_groups(self, conf, type, name):
    """
        This function fetches the description and invoke
        the parsing of group members.
        :param conf: configuration.
        :param type: type of group.
        :param name: name of group.
        :return: generated configuration dictionary.
        """
    a_lst = ['name', 'description']
    group = self.parse_attr(conf, a_lst)
    key = self.get_key(type)
    r_sub = {key[0]: self.parse_address_port_lst(conf, name, key[1])}
    group.update(r_sub)
    return group