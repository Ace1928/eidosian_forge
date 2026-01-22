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
@cliutils.arg('--state', metavar='<state>', default='available', help='Indicate which state to assign the share group snapshot. Options include available, error, creating, deleting, error_deleting. If no state is provided, available will be used.')
@cliutils.arg('share_group_snapshot', metavar='<share_group_snapshot>', help='Name or ID of the share group snapshot.')
@cliutils.service_type('sharev2')
def do_share_group_snapshot_reset_state(cs, args):
    """Explicitly update the state of a share group snapshot

    (Admin only).
    """
    sg_snapshot = _find_share_group_snapshot(cs, args.share_group_snapshot)
    cs.share_group_snapshots.reset_state(sg_snapshot, args.state)