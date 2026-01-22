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
@cliutils.arg('--name', metavar='<name>', help='Optional share group name. (Default=None)', default=None)
@cliutils.arg('--description', metavar='<description>', help='Optional share group description. (Default=None)', default=None)
@cliutils.arg('--share-types', '--share_types', metavar='<share_types>', type=str, default=None, action='single_alias', help='Comma-separated list of share types. (Default=None)')
@cliutils.arg('--share-group-type', '--share_group_type', '--type', metavar='<share_group_type>', type=str, default=None, action='single_alias', help='Share group type name or ID of the share group to be created. (Default=None)')
@cliutils.arg('--share-network', '--share_network', metavar='<share_network>', type=str, default=None, action='single_alias', help='Specify share network name or id.')
@cliutils.arg('--source-share-group-snapshot', '--source_share_group_snapshot', metavar='<source_share_group_snapshot>', type=str, action='single_alias', help='Optional share group snapshot name or ID to create the share group from. (Default=None)', default=None)
@cliutils.arg('--availability-zone', '--availability_zone', '--az', default=None, action='single_alias', metavar='<availability-zone>', help='Optional availability zone in which group should be created. (Default=None)')
@cliutils.arg('--wait', action='store_true', default=False, help='Wait for share group to create')
@cliutils.service_type('sharev2')
def do_share_group_create(cs, args):
    """Creates a new share group."""
    share_types = []
    if args.share_types:
        s_types = args.share_types.split(',')
        for s_type in s_types:
            share_type = _find_share_type(cs, s_type)
            share_types.append(share_type)
    share_group_type = None
    if args.share_group_type:
        share_group_type = _find_share_group_type(cs, args.share_group_type)
    share_network = None
    if args.share_network:
        share_network = _find_share_network(cs, args.share_network)
    share_group_snapshot = None
    if args.source_share_group_snapshot:
        share_group_snapshot = _find_share_group_snapshot(cs, args.source_share_group_snapshot)
    kwargs = {'share_group_type': share_group_type, 'share_types': share_types or None, 'name': args.name, 'description': args.description, 'availability_zone': args.availability_zone, 'source_share_group_snapshot': share_group_snapshot, 'share_network': share_network}
    share_group = cs.share_groups.create(**kwargs)
    if args.wait:
        try:
            share_group = _wait_for_resource_status(cs, share_group, resource_type='share_group', expected_status='available')
        except exceptions.CommandError as e:
            print(e, file=sys.stderr)
    _print_share_group(cs, share_group)