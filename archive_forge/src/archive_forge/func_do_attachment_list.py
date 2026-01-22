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
@api_versions.wraps('3.27')
@utils.arg('--all-tenants', dest='all_tenants', metavar='<0|1>', nargs='?', type=int, const=1, default=utils.env('ALL_TENANTS', default=0), help='Shows details for all tenants. Admin only.')
@utils.arg('--volume-id', metavar='<volume-id>', default=None, help='Filters results by a volume ID. Default=None. %s' % FILTER_DEPRECATED)
@utils.arg('--status', metavar='<status>', default=None, help='Filters results by a status. Default=None. %s' % FILTER_DEPRECATED)
@utils.arg('--marker', metavar='<marker>', default=None, help='Begin returning attachments that appear later in attachment list than that represented by this id. Default=None.')
@utils.arg('--limit', metavar='<limit>', default=None, help='Maximum number of attachments to return. Default=None.')
@utils.arg('--sort', metavar='<key>[:<direction>]', default=None, help='Comma-separated list of sort keys and directions in the form of <key>[:<asc|desc>]. Valid keys: %s. Default=None.' % ', '.join(base.SORT_KEY_VALUES))
@utils.arg('--tenant', type=str, dest='tenant', nargs='?', metavar='<tenant>', help='Display information from single tenant (Admin only).')
@utils.arg('--filters', action=AppendFilters, type=str, nargs='*', start_version='3.33', metavar='<key=value>', default=None, help="Filter key and value pairs. Please use 'cinder list-filters' to check enabled filters from server. Use 'key~=value' for inexact filtering if the key supports. Default=None.")
def do_attachment_list(cs, args):
    """Lists all attachments."""
    search_opts = {'all_tenants': 1 if args.tenant else args.all_tenants, 'project_id': args.tenant, 'status': args.status, 'volume_id': args.volume_id}
    if AppendFilters.filters:
        search_opts.update(shell_utils.extract_filters(AppendFilters.filters))
    attachments = cs.attachments.list(search_opts=search_opts, marker=args.marker, limit=args.limit, sort=args.sort)
    for attachment in attachments:
        setattr(attachment, 'server_id', getattr(attachment, 'instance', None))
    columns = ['ID', 'Volume ID', 'Status', 'Server ID']
    if args.sort:
        sortby_index = None
    else:
        sortby_index = 0
    shell_utils.print_list(attachments, columns, sortby_index=sortby_index)
    AppendFilters.filters = []