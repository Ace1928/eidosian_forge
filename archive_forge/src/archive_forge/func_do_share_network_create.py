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
@api_versions.wraps('2.26')
@cliutils.arg('--neutron-net-id', '--neutron-net_id', '--neutron_net_id', '--neutron_net-id', metavar='<neutron-net-id>', default=None, action='single_alias', help='Neutron network ID. Used to set up network for share servers.')
@cliutils.arg('--neutron-subnet-id', '--neutron-subnet_id', '--neutron_subnet_id', '--neutron_subnet-id', metavar='<neutron-subnet-id>', default=None, action='single_alias', help='Neutron subnet ID. Used to set up network for share servers. This subnet should belong to specified neutron network.')
@cliutils.arg('--name', metavar='<name>', default=None, help='Share network name.')
@cliutils.arg('--description', metavar='<description>', default=None, help='Share network description.')
@cliutils.arg('--availability-zone', '--availability_zone', '--az', metavar='<availability_zone>', default=None, action='single_alias', help="Availability zone in which the subnet should be created. Share networks can have one or more subnets in different availability zones when the driver is operating with 'driver_handles_share_servers' extra_spec set to True. Available only for microversion >= 2.51. (Default=None)")
def do_share_network_create(cs, args):
    """Create a share network to export shares to."""
    values = {'neutron_net_id': args.neutron_net_id, 'neutron_subnet_id': args.neutron_subnet_id, 'name': args.name, 'description': args.description}
    if cs.api_version >= api_versions.APIVersion('2.51'):
        values['availability_zone'] = args.availability_zone
    elif args.availability_zone:
        raise exceptions.CommandError('Creating share networks with a given az is only available with manila API version >= 2.51')
    share_network = cs.share_networks.create(**values)
    info = share_network._info.copy()
    cliutils.print_dict(info)