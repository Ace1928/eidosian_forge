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
@api_versions.wraps('2.45')
@cliutils.arg('access_id', metavar='<access_id>', help='ID of the NAS share access rule.')
def do_access_show(cs, args):
    """Show details about a NAS share access rule."""
    access = cs.share_access_rules.get(args.access_id)
    view_data = access._info.copy()
    cliutils.print_dict(view_data)