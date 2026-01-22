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
@cliutils.arg('share_group', metavar='<share_group>', help='Name or ID of the share group.')
@cliutils.service_type('sharev2')
def do_share_group_show(cs, args):
    """Show details about a share group."""
    share_group = _find_share_group(cs, args.share_group)
    _print_share_group(cs, share_group)