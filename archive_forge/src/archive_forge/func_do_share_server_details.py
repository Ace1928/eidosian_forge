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
@cliutils.arg('id', metavar='<id>', type=str, help='ID of share server.')
def do_share_server_details(cs, args):
    """Show share server details (Admin only)."""
    details = cs.share_servers.details(args.id)
    cliutils.print_dict(details._info)