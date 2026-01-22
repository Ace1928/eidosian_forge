import argparse
import collections
import copy
import os
from oslo_utils import strutils
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3 import availability_zones
@utils.arg('snapshot', metavar='<snapshot>', nargs='+', help='Name or ID of snapshot to modify.')
@utils.arg('--state', metavar='<state>', default='available', help='The state to assign to the snapshot. Valid values are "available", "error", "creating", "deleting", and "error_deleting". NOTE: This command simply changes the state of the Snapshot in the DataBase with no regard to actual status, exercise caution when using. Default=available.')
def do_snapshot_reset_state(cs, args):
    """Explicitly updates the snapshot state."""
    failure_count = 0
    single = len(args.snapshot) == 1
    for snapshot in args.snapshot:
        try:
            shell_utils.find_volume_snapshot(cs, snapshot).reset_state(args.state)
        except Exception as e:
            failure_count += 1
            msg = 'Reset state for snapshot %s failed: %s' % (snapshot, e)
            if not single:
                print(msg)
    if failure_count == len(args.snapshot):
        if not single:
            msg = 'Unable to reset the state for any of the specified snapshots.'
        raise exceptions.CommandError(msg)