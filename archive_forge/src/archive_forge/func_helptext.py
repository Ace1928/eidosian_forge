from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
from googlecloudsdk.api_lib.edge_cloud.networking import utils
from googlecloudsdk.calliope import arg_parsers
def helptext(verb, prep):
    return '{} the comma-separated list of CIDRs {} the set of range advertisements.'.format(verb, prep)