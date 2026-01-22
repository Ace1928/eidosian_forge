from __future__ import absolute_import, division, print_function
import re
import socket
import sys
import traceback
from ansible.module_utils.basic import env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible.module_utils.connection import exec_command, ConnectionError
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import NetconfConnection
def get_xml_line(xml_list, index):
    """get xml specified line valid string data"""
    ele = None
    while xml_list and (not ele):
        if index >= 0 and index >= len(xml_list):
            return None
        if index < 0 and abs(index) > len(xml_list):
            return None
        ele = xml_list[index]
        if not ele.replace(' ', ''):
            xml_list.pop(index)
            ele = None
    return ele