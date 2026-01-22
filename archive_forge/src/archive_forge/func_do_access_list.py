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
@api_versions.wraps('2.33')
@cliutils.arg('share', metavar='<share>', help='Name or ID of the share.')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "access_type,access_to".')
@cliutils.arg('--metadata', type=str, nargs='*', metavar='<key=value>', help='Filters results by a metadata key and value. OPTIONAL: Default=None. Available only for microversion >= 2.45', default=None)
def do_access_list(cs, args):
    """Show access list for share."""
    list_of_keys = ['id', 'access_type', 'access_to', 'access_level', 'state', 'access_key', 'created_at', 'updated_at']
    share = _find_share(cs, args.share)
    if cs.api_version < api_versions.APIVersion('2.45'):
        if getattr(args, 'metadata'):
            raise exceptions.CommandError('Filtering access rules by metadata is supported only beyond API version 2.45')
        access_list = share.access_list()
    else:
        access_list = cs.share_access_rules.access_list(share, {'metadata': _extract_metadata(args)})
    if args.columns is not None:
        list_of_keys = _split_columns(columns=args.columns)
    cliutils.print_list(access_list, list_of_keys)