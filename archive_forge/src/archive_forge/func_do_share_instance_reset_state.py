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
@cliutils.arg('instance', metavar='<instance>', help='Name or ID of the share instance to modify.')
@cliutils.arg('--state', metavar='<state>', default='available', help='Indicate which state to assign the instance. Options include available, error, creating, deleting, error_deleting, migrating,migrating_to. If no state is provided, available will be used.')
@api_versions.wraps('2.3')
def do_share_instance_reset_state(cs, args):
    """Explicitly update the state of a share instance (Admin only)."""
    instance = _find_share_instance(cs, args.instance)
    instance.reset_state(args.state)