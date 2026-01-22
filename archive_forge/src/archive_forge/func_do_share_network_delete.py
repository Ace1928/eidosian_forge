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
@cliutils.arg('share_network', metavar='<share-network>', nargs='+', help='Name or ID of share network(s) to be deleted.')
def do_share_network_delete(cs, args):
    """Delete one or more share networks."""
    failure_count = 0
    for share_network in args.share_network:
        try:
            share_ref = _find_share_network(cs, share_network)
            cs.share_networks.delete(share_ref)
        except Exception as e:
            failure_count += 1
            print('Delete for share network %s failed: %s' % (share_network, e), file=sys.stderr)
    if failure_count == len(args.share_network):
        raise exceptions.CommandError('Unable to delete any of the specified share networks.')