import argparse
import collections
import os
from oslo_utils import strutils
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3.shell_base import *  # noqa
from cinderclient.v3.shell_base import CheckSizeArgForCreate
@api_versions.wraps('3.38')
@utils.arg('group', metavar='<group>', help='Name or ID of the group.')
@utils.arg('--allow-attached-volume', action='store_true', default=False, help='Allows or disallows group with attached volumes to be failed over.')
@utils.arg('--secondary-backend-id', metavar='<secondary_backend_id>', help='Secondary backend id. Default=None.')
def do_group_failover_replication(cs, args):
    """Fails over replication for group."""
    shell_utils.find_group(cs, args.group).failover_replication(allow_attached_volume=args.allow_attached_volume, secondary_backend_id=args.secondary_backend_id)