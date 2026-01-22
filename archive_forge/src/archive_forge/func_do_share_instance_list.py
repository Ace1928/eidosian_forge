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
@cliutils.arg('--share-id', '--share_id', metavar='<share_id>', default=None, action='single_alias', help='Filter results by share ID.')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "id,host,status".')
@cliutils.arg('--export-location', '--export_location', metavar='<export_location>', type=str, default=None, action='single_alias', help='ID or path of the share instance export location. Available only for microversion >= 2.35.')
@api_versions.wraps('2.3')
def do_share_instance_list(cs, args):
    """List share instances (Admin only)."""
    share = _find_share(cs, args.share_id) if args.share_id else None
    list_of_keys = ['ID', 'Share ID', 'Host', 'Status', 'Availability Zone', 'Share Network ID', 'Share Server ID', 'Share Type ID']
    if args.columns is not None:
        list_of_keys = _split_columns(columns=args.columns)
    if share:
        instances = cs.shares.list_instances(share)
    elif cs.api_version.matches(api_versions.APIVersion('2.35'), api_versions.APIVersion()):
        instances = cs.share_instances.list(args.export_location)
    else:
        if args.export_location:
            raise exceptions.CommandError('Filtering by export location is only available with manila API version >= 2.35')
        instances = cs.share_instances.list()
    cliutils.print_list(instances, list_of_keys)