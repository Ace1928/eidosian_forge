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
@utils.arg('host', metavar='<host>', help='Cinder host on which to list manageable volumes; takes the form: host@backend-name#pool')
@utils.arg('--detailed', metavar='<detailed>', default=True, help='Returned detailed information (default true).')
@utils.arg('--marker', metavar='<marker>', default=None, help='Begin returning volumes that appear later in the volume list than that represented by this volume id. Default=None.')
@utils.arg('--limit', metavar='<limit>', default=None, help='Maximum number of volumes to return. Default=None.')
@utils.arg('--offset', metavar='<offset>', default=None, help='Number of volumes to skip after marker. Default=None.')
@utils.arg('--sort', metavar='<key>[:<direction>]', default=None, help='Comma-separated list of sort keys and directions in the form of <key>[:<asc|desc>]. Valid keys: %s. Default=None.' % ', '.join(base.SORT_KEY_VALUES))
def do_manageable_list(cs, args):
    """Lists all manageable volumes."""
    detailed = strutils.bool_from_string(args.detailed)
    volumes = cs.volumes.list_manageable(host=args.host, detailed=detailed, marker=args.marker, limit=args.limit, offset=args.offset, sort=args.sort)
    columns = ['reference', 'size', 'safe_to_manage']
    if detailed:
        columns.extend(['reason_not_safe', 'cinder_id', 'extra_info'])
    shell_utils.print_list(volumes, columns, sortby_index=None)