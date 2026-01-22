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
@api_versions.wraps('2.32')
@cliutils.arg('snapshot', metavar='<snapshot>', help='Name or ID of the share snapshot to deny access to.')
@cliutils.arg('id', metavar='<id>', nargs='+', help='ID(s) of the access rule(s) to be deleted.')
def do_snapshot_access_deny(cs, args):
    """Deny access to a snapshot."""
    failure_count = 0
    snapshot = _find_share_snapshot(cs, args.snapshot)
    for access_id in args.id:
        try:
            snapshot.deny(access_id)
        except Exception as e:
            failure_count += 1
            print('Failed to remove rule %(access)s: %(reason)s.' % {'access': access_id, 'reason': e}, file=sys.stderr)
    if failure_count == len(args.id):
        raise exceptions.CommandError('Unable to delete any of the specified snapshot rules.')