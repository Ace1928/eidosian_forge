from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from datetime import datetime
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import (
from ipaddress import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip_interface
from ..module_utils.teem import send_teem
def decode_address_from_string(self, record):
    matches = self._rd_net_prefix_ptrn.match(record)
    if matches:
        value = record.split(self._separator)[1].strip().strip('"')
        addr = '{0}%{1}/{2}'.format(matches.group('addr'), matches.group('rd'), matches.group('prefix'))
        result = dict(name=addr, data=value)
        return result
    matches = self._net_prefix_pattern.match(record)
    if matches:
        key = u'{0}/{1}'.format(matches.group('addr'), matches.group('prefix'))
        addr = ip_network(key)
        value = record.split(self._separator)[1].strip().strip('"')
        result = dict(name=str(addr), data=value)
        return result
    matches = self._net_pattern.match(record)
    if matches:
        key = u'{0}'.format(matches.group('addr'))
        addr = ip_network(key)
        if len(record.split(self._separator)) > 1:
            value = record.split(self._separator)[1].strip().strip('"')
            result = dict(name=str(addr), data=value)
            return result
        return str(record)
    matches = self._rd_host_ptrn.match(record)
    if matches:
        host = ip_interface(u'{0}'.format(matches.group('addr')))
        addr = '{0}%{1}/{2}'.format(matches.group('addr'), matches.group('rd'), str(host.network.prefixlen))
        value = record.split(self._separator)[1].strip().strip('"')
        result = dict(name=addr, data=value)
        return result
    matches = self._host_pattern.match(record)
    if matches:
        key = matches.group('addr')
        addr = ip_interface(u'{0}'.format(str(key)))
        if len(record.split(self._separator)) > 1:
            value = record.split(self._separator)[1].strip().strip('"')
            result = dict(name=str(addr), data=value)
            return result
        return str(record)
    raise F5ModuleError('The value "{0}" is not an address'.format(record))