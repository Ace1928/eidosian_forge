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
@cliutils.arg('replica', metavar='<replica>', help='Name or ID of the share replica.')
@cliutils.arg('export_location', metavar='<export_location>', help='ID of the share replica export location.')
def do_share_replica_export_location_show(cs, args):
    """Show details of a share replica's export location."""
    replica = _find_share_replica(cs, args.replica)
    export_location = cs.share_replica_export_locations.get(replica, args.export_location)
    view_data = export_location._info.copy()
    cliutils.print_dict(view_data)