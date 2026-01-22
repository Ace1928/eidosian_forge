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
@cliutils.arg('share', metavar='<share>', help='Name or ID of the share to modify.')
@cliutils.arg('--task-state', '--task_state', '--state', metavar='<task_state>', default='None', action='single_alias', required=False, help='Indicate which task state to assign the share. Options include migration_starting, migration_in_progress, migration_completing, migration_success, migration_error, migration_cancelled, migration_driver_in_progress, migration_driver_phase1_done, data_copying_starting, data_copying_in_progress, data_copying_completing, data_copying_completed, data_copying_cancelled, data_copying_error. If no value is provided, None will be used.')
@api_versions.wraps('2.22')
def do_reset_task_state(cs, args):
    """Explicitly update the task state of a share

    (Admin only, Experimental).
    """
    state = args.task_state
    if args.task_state == 'None':
        state = None
    share = _find_share(cs, args.share)
    share.reset_task_state(state)