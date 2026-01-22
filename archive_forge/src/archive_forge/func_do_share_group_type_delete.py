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
@cliutils.arg('id', metavar='<id>', help='Name or ID of the share group type to delete.')
@cliutils.service_type('sharev2')
def do_share_group_type_delete(cs, args):
    """Delete a specific share group type (Admin only)."""
    share_group_type = _find_share_group_type(cs, args.id)
    cs.share_group_types.delete(share_group_type)