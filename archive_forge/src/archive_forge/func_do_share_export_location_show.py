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
@api_versions.wraps('2.9')
@cliutils.arg('share', metavar='<share>', help='Name or ID of the share.')
@cliutils.arg('export_location', metavar='<export_location>', help='ID of the share export location.')
def do_share_export_location_show(cs, args):
    """Show export location of the share."""
    share = _find_share(cs, args.share)
    export_location = cs.share_export_locations.get(share, args.export_location)
    view_data = export_location._info.copy()
    cliutils.print_dict(view_data)