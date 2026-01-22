import base64
import binascii
import os
import re
import shlex
from oslo_serialization import jsonutils
from oslo_utils import netutils
from urllib import parse
from urllib import request
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import cliutils as utils
from zunclient import exceptions as exc
from zunclient.i18n import _
def parse_nets(ns):
    err_msg = "Invalid nets argument '%s'. nets arguments must be of the form --nets <network=network, v4-fixed-ip=ip-addr,v6-fixed-ip=ip-addr, port=port-uuid>, with only one of network, or port specified."
    nets = []
    for net_str in ns:
        keys = ['network', 'port', 'v4-fixed-ip', 'v6-fixed-ip']
        net_info = {}
        for kv_str in net_str.split(','):
            try:
                k, v = kv_str.split('=', 1)
                k = k.strip()
                v = v.strip()
            except ValueError:
                raise apiexec.CommandError(err_msg % net_str)
            if k in keys:
                if net_info.get(k):
                    raise apiexec.CommandError(err_msg % net_str)
                net_info[k] = v
            else:
                raise apiexec.CommandError(err_msg % net_str)
        if net_info.get('v4-fixed-ip') and (not netutils.is_valid_ipv4(net_info['v4-fixed-ip'])):
            raise apiexec.CommandError('Invalid ipv4 address.')
        if net_info.get('v6-fixed-ip') and (not netutils.is_valid_ipv6(net_info['v6-fixed-ip'])):
            raise apiexec.CommandError('Invalid ipv6 address.')
        if bool(net_info.get('network')) == bool(net_info.get('port')):
            raise apiexec.CommandError(err_msg % net_str)
        nets.append(net_info)
    return nets