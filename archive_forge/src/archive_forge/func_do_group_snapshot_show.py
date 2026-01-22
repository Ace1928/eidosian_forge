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
@api_versions.wraps('3.14')
@utils.arg('group_snapshot', metavar='<group_snapshot>', help='Name or ID of group snapshot.')
def do_group_snapshot_show(cs, args):
    """Shows group snapshot details."""
    info = dict()
    group_snapshot = shell_utils.find_group_snapshot(cs, args.group_snapshot)
    info.update(group_snapshot._info)
    info.pop('links', None)
    shell_utils.print_dict(info)