from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_rules.firewall_rules import (
def parse_p2p(self, conf):
    """
        This function forms the regex to fetch the 'p2p' with in
        'rules'
        :param conf: configuration data.
        :return: generated rule list configuration.
        """
    a_lst = []
    applications = findall("p2p (?:\\'*)(\\d+)(?:\\'*)", conf, M)
    if applications:
        app_lst = []
        for r in set(applications):
            obj = {'application': r.strip("'")}
            app_lst.append(obj)
        a_lst = sorted(app_lst, key=lambda i: i['application'])
    return a_lst