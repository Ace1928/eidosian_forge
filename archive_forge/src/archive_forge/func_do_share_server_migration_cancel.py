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
@cliutils.arg('share_server_id', metavar='<share_server_id>', help='ID of share server to complete migration.')
@api_versions.wraps('2.57')
@api_versions.experimental_api
def do_share_server_migration_cancel(cs, args):
    """Cancels migration of a given share server when copying

    (Admin only, Experimental).
    """
    share_server = _find_share_server(cs, args.share_server_id)
    share_server.migration_cancel()