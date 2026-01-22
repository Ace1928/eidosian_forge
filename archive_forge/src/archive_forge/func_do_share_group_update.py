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
@cliutils.arg('share_group', metavar='<share_group>', help='Name or ID of the share group to update.')
@cliutils.arg('--name', metavar='<name>', default=None, help='Optional new name for the share group. (Default=None)')
@cliutils.arg('--description', metavar='<description>', help='Optional share group description. (Default=None)', default=None)
@cliutils.service_type('sharev2')
def do_share_group_update(cs, args):
    """Update a share group."""
    kwargs = {}
    if args.name is not None:
        kwargs['name'] = args.name
    if args.description is not None:
        kwargs['description'] = args.description
    if not kwargs:
        msg = 'Must supply name and/or description'
        raise exceptions.CommandError(msg)
    share_group = _find_share_group(cs, args.share_group)
    share_group = cs.share_groups.update(share_group, **kwargs)
    _print_share_group(cs, share_group)