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
@cliutils.arg('--all', dest='all', action='store_true', default=False, help='Display all share group types (Admin only).')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "id,name".')
@cliutils.service_type('sharev2')
def do_share_group_type_list(cs, args):
    """Print a list of available 'share group types'."""
    sg_types = cs.share_group_types.list(show_all=args.all)
    default = None
    if sg_types and (not hasattr(sg_types[0], 'is_default')):
        if args.columns and 'is_default' in args.columns or args.columns is None:
            default = cs.share_group_types.get()
    _print_share_group_type_list(sg_types, default_share_group_type=default, columns=args.columns)