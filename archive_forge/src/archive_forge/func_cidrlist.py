from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
from googlecloudsdk.api_lib.edge_cloud.networking import utils
from googlecloudsdk.calliope import arg_parsers
def cidrlist(argstr):
    split = argstr.split(',')
    parsed = map(ipaddress.ip_network, split)
    retlist = sorted(parsed)
    retset = set(retlist)
    if len(retlist) != len(retset):
        raise ValueError('CIDR list contained duplicates.')
    return retlist