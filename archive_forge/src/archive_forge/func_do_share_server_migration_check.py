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
@cliutils.arg('share_server_id', metavar='<share_server_id>', help='ID of the share server to check if the migration is possible.')
@cliutils.arg('host', metavar='<host@backend>', help="Destination to migrate the share server to. Use the format '<node_hostname>@<backend_name>'.")
@cliutils.arg('--preserve-snapshots', '--preserve_snapshots', action='single_alias', metavar='<True|False>', choices=['True', 'False'], required=True, help='Set to True if snapshots must be preserved at the migration destination.')
@cliutils.arg('--writable', metavar='<True|False>', choices=['True', 'False'], required=True, help='Set to True if shares associated with the share server must be writable through the first phase of the migration.')
@cliutils.arg('--nondisruptive', metavar='<True|False>', choices=['True', 'False'], required=True, help='Set to True if migration must be non disruptive to clients that are using the shares associated with the share server through both phases of the migration.')
@cliutils.arg('--new_share_network', '--new-share-network', metavar='<new_share_network>', action='single_alias', required=False, help='New share network to migrate to. Optional, default=None.', default=None)
@api_versions.wraps('2.57')
@api_versions.experimental_api
def do_share_server_migration_check(cs, args):
    """Check migration compatibility for a share server with desired properties

    (Admin only, Experimental).
    """
    share_server = _find_share_server(cs, args.share_server_id)
    new_share_net_id = None
    if args.new_share_network:
        share_net = _find_share_network(cs, args.new_share_network)
        new_share_net_id = share_net.id
    result = share_server.migration_check(args.host, args.writable, args.nondisruptive, args.preserve_snapshots, new_share_net_id)
    cliutils.print_dict(result)