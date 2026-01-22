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
@utils.arg('snapshot', metavar='<snapshot>', help='Name or ID of snapshot.')
@utils.arg('name', nargs='?', metavar='<name>', help='New name for snapshot.')
@utils.arg('--description', metavar='<description>', default=None, help='Snapshot description. Default=None.')
@utils.arg('--display-description', help=argparse.SUPPRESS)
@utils.arg('--display_description', help=argparse.SUPPRESS)
def do_snapshot_rename(cs, args):
    """Renames a snapshot."""
    kwargs = {}
    if args.name is not None:
        kwargs['name'] = args.name
    if args.description is not None:
        kwargs['description'] = args.description
    elif args.display_description is not None:
        kwargs['description'] = args.display_description
    if not any(kwargs):
        msg = 'Must supply either name or description.'
        raise exceptions.ClientException(code=1, message=msg)
    shell_utils.find_volume_snapshot(cs, args.snapshot).update(**kwargs)
    print("Request to rename snapshot '%s' has been accepted." % args.snapshot)