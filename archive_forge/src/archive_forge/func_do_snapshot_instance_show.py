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
@api_versions.wraps('2.19')
@cliutils.arg('snapshot_instance', metavar='<snapshot_instance>', help='ID of the share snapshot instance.')
def do_snapshot_instance_show(cs, args):
    """Show details about a share snapshot instance."""
    snapshot_instance = _find_share_snapshot_instance(cs, args.snapshot_instance)
    export_locations = cs.share_snapshot_instance_export_locations.list(snapshot_instance)
    snapshot_instance._info['export_locations'] = export_locations
    _print_share_snapshot(cs, snapshot_instance)