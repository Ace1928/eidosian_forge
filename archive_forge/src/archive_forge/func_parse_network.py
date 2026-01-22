from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.ospfv2.ospfv2 import (
def parse_network(self, conf):
    """
        This function forms the regex to fetch the 'network'
        :param conf: configuration data
        :return: generated rule list configuration
        """
    a_lst = []
    applications = findall('network (.+)', conf, M)
    if applications:
        app_lst = []
        for r in set(applications):
            obj = {'address': r.strip("'")}
            app_lst.append(obj)
        a_lst = sorted(app_lst, key=lambda i: i['address'])
    return a_lst