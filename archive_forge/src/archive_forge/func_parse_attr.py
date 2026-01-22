from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.ospfv2.ospfv2 import (
def parse_attr(self, conf, attr_list, match=None):
    """
        This function peforms the following:
        - Form the regex to fetch the required attribute config.
        - Type cast the output in desired format.
        :param conf: configuration.
        :param attr_list: list of attributes.
        :param match: parent node/attribute name.
        :return: generated config dictionary.
        """
    config = {}
    for attrib in attr_list:
        regex = self.map_regex(attrib)
        if match:
            regex = match.replace('_', '-') + ' ' + regex
        if conf:
            if self.is_bool(attrib):
                out = conf.find(attrib.replace('_', '-'))
                dis = conf.find(attrib.replace('_', '-') + " 'disable'")
                if match:
                    if attrib == 'set' and conf.find(match) >= 1:
                        config[attrib] = True
                    en = conf.find(match + " 'enable'")
                if out >= 1:
                    if dis >= 1:
                        config[attrib] = False
                    else:
                        config[attrib] = True
                elif match and en >= 1:
                    config[attrib] = True
            else:
                out = search('^.*' + regex + ' (.+)', conf, M)
                if out:
                    val = out.group(1).strip("'")
                    if self.is_num(attrib):
                        val = int(val)
                    config[attrib] = val
    return config