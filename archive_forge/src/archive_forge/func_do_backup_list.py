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
@utils.arg('--all-tenants', metavar='<all_tenants>', nargs='?', type=int, const=1, default=0, help='Shows details for all tenants. Admin only.')
@utils.arg('--all_tenants', nargs='?', type=int, const=1, help=argparse.SUPPRESS)
@utils.arg('--name', metavar='<name>', default=None, help='Filters results by a name. Default=None.')
@utils.arg('--status', metavar='<status>', default=None, help='Filters results by a status. Default=None.')
@utils.arg('--volume-id', metavar='<volume-id>', default=None, help='Filters results by a volume ID. Default=None.')
@utils.arg('--volume_id', help=argparse.SUPPRESS)
@utils.arg('--marker', metavar='<marker>', default=None, help='Begin returning backups that appear later in the backup list than that represented by this id. Default=None.')
@utils.arg('--limit', metavar='<limit>', default=None, help='Maximum number of backups to return. Default=None.')
@utils.arg('--sort', metavar='<key>[:<direction>]', default=None, help='Comma-separated list of sort keys and directions in the form of <key>[:<asc|desc>]. Valid keys: %s. Default=None.' % ', '.join(base.SORT_KEY_VALUES))
def do_backup_list(cs, args):
    """Lists all backups."""
    search_opts = {'all_tenants': args.all_tenants, 'name': args.name, 'status': args.status, 'volume_id': args.volume_id}
    backups = cs.backups.list(search_opts=search_opts, marker=args.marker, limit=args.limit, sort=args.sort)
    shell_utils.translate_volume_snapshot_keys(backups)
    columns = ['ID', 'Volume ID', 'Status', 'Name', 'Size', 'Object Count', 'Container']
    if args.sort:
        sortby_index = None
    else:
        sortby_index = 0
    shell_utils.print_list(backups, columns, sortby_index=sortby_index)