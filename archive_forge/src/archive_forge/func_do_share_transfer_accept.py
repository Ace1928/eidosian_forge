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
@api_versions.wraps('2.77')
@cliutils.arg('transfer', metavar='<transfer>', help='ID of transfer to accept.')
@cliutils.arg('auth_key', metavar='<auth_key>', help='Authentication key of transfer to accept.')
@cliutils.arg('--clear-rules', '--clear_rules', dest='clear_rules', action='store_true', default=False, help='Whether manila should clean up the access rules after the transfer is complete. (Default=False)')
def do_share_transfer_accept(cs, args):
    """Accepts a share transfer."""
    cs.transfers.accept(args.transfer, args.auth_key, clear_access_rules=args.clear_rules)