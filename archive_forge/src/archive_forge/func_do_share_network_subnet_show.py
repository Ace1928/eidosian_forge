from operator import xor
import os
import re
import sys
import time
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
import manilaclient.v2.shares
@cliutils.arg('share_network', metavar='<share-network>', help='Name or ID of share network(s) to which the subnet belongs.')
@cliutils.arg('share_network_subnet', metavar='<share-network-subnet>', help='Share network subnet ID to show.')
def do_share_network_subnet_show(cs, args):
    """Show share network subnet."""
    share_network = _find_share_network(cs, args.share_network)
    share_network_subnet = cs.share_network_subnets.get(share_network.id, args.share_network_subnet)
    view_data = share_network_subnet._info.copy()
    cliutils.print_dict(view_data)