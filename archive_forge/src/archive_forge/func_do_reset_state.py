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
@utils.arg('volume', metavar='<volume>', nargs='+', help='Name or ID of volume to modify.')
@utils.arg('--state', metavar='<state>', default=None, help='The state to assign to the volume. Valid values are "available", "error", "creating", "deleting", "in-use", "attaching", "detaching", "error_deleting" and "maintenance". NOTE: This command simply changes the state of the Volume in the DataBase with no regard to actual status, exercise caution when using. Default=None, that means the state is unchanged.')
@utils.arg('--attach-status', metavar='<attach-status>', default=None, help='The attach status to assign to the volume in the DataBase, with no regard to the actual status. Valid values are "attached" and "detached". Default=None, that means the status is unchanged.')
@utils.arg('--reset-migration-status', action='store_true', help='Clears the migration status of the volume in the DataBase that indicates the volume is source or destination of volume migration, with no regard to the actual status.')
def do_reset_state(cs, args):
    """Explicitly updates the volume state in the Cinder database.

    Note that this does not affect whether the volume is actually attached to
    the Nova compute host or instance and can result in an unusable volume.
    Being a database change only, this has no impact on the true state of the
    volume and may not match the actual state. This can render a volume
    unusable in the case of change to the 'available' state.
    """
    failure_flag = False
    migration_status = 'none' if args.reset_migration_status else None
    if not (args.state or args.attach_status or migration_status):
        args.state = 'available'
    for volume in args.volume:
        try:
            utils.find_volume(cs, volume).reset_state(args.state, args.attach_status, migration_status)
        except Exception as e:
            failure_flag = True
            msg = 'Reset state for volume %s failed: %s' % (volume, e)
            print(msg)
    if failure_flag:
        msg = 'Unable to reset the state for the specified volume(s).'
        raise exceptions.CommandError(msg)