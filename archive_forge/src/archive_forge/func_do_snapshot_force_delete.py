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
@cliutils.arg('snapshot', metavar='<snapshot>', nargs='+', help='Name or ID of the snapshot(s) to force delete.')
@cliutils.arg('--wait', action='store_true', help='Wait for snapshot to delete')
@cliutils.service_type('sharev2')
def do_snapshot_force_delete(cs, args):
    """Attempt force-deletion of one or more snapshots.

    Regardless of the state (Admin only).
    """
    failure_count = 0
    snapshots_to_delete = []
    for snapshot in args.snapshot:
        try:
            snapshot_ref = _find_share_snapshot(cs, snapshot)
            snapshots_to_delete.append(snapshot_ref)
            snapshot_ref.force_delete()
        except Exception as e:
            failure_count += 1
            print('Delete for snapshot %s failed: %s' % (snapshot, e), file=sys.stderr)
    if failure_count == len(args.snapshot):
        raise exceptions.CommandError('Unable to force delete any of the specified snapshots.')
    if args.wait:
        for snapshot in snapshots_to_delete:
            try:
                _wait_for_resource_status(cs, snapshot, resource_type='snapshot', expected_status='deleted')
            except exceptions.CommandError as e:
                print(e, file=sys.stderr)