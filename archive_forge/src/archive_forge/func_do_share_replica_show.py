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
@api_versions.wraps('2.47')
@cliutils.arg('replica', metavar='<replica>', help='ID of the share replica.')
def do_share_replica_show(cs, args):
    """Show details about a replica."""
    replica = cs.share_replicas.get(args.replica)
    export_locations = cs.share_replica_export_locations.list(replica)
    replica._info['export_locations'] = export_locations
    _print_share_replica(cs, replica)