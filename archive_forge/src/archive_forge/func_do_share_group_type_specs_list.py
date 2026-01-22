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
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "id,name".')
@cliutils.service_type('sharev2')
def do_share_group_type_specs_list(cs, args):
    """Print a list of 'share group types specs' (Admin Only)."""
    sg_types = cs.share_group_types.list()
    _print_type_and_extra_specs_list(sg_types, columns=args.columns)