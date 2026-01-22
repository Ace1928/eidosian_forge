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
@cliutils.arg('share_network', metavar='<share-network>', help='Share network name or ID.')
@cliutils.arg('--neutron-net-id', '--neutron-net_id', '--neutron_net_id', '--neutron_net-id', metavar='<neutron-net-id>', default=None, action='single_alias', help='Neutron network ID. Used to set up network for share servers. Optional, Default = None.')
@cliutils.arg('--neutron-subnet-id', '--neutron-subnet_id', '--neutron_subnet_id', '--neutron_subnet-id', metavar='<neutron-subnet-id>', default=None, action='single_alias', help='Neutron subnet ID. Used to set up network for share servers. This subnet should belong to specified neutron network. Optional, Default = None.')
@cliutils.arg('--availability-zone', '--availability_zone', '--az', default=None, action='single_alias', metavar='<availability-zone>', help='Optional availability zone that the subnet is available within (Default=None). If None, the subnet will be considered as being available across all availability zones.')
@cliutils.arg('--reset', metavar='<True|False>', choices=['True', 'False'], required=False, default=False, help='Reset and start again the check operation.(Optional, Default=False)')
@api_versions.wraps('2.70')
def do_share_network_subnet_create_check(cs, args):
    """Check if a new subnet can be added to a share network."""
    if xor(bool(args.neutron_net_id), bool(args.neutron_subnet_id)):
        raise exceptions.CommandError('Both neutron_net_id and neutron_subnet_id should be specified. Alternatively, neither of them should be specified.')
    share_network = _find_share_network(cs, args.share_network)
    values = {'neutron_net_id': args.neutron_net_id, 'neutron_subnet_id': args.neutron_subnet_id, 'availability_zone': args.availability_zone, 'reset_operation': args.reset}
    subnet_create_check = cs.share_networks.share_network_subnet_create_check(share_network.id, **values)
    cliutils.print_dict(subnet_create_check[1])