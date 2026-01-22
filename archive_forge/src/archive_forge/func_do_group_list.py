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
@api_versions.wraps('3.13')
@utils.arg('--all-tenants', dest='all_tenants', metavar='<0|1>', nargs='?', type=int, const=1, default=utils.env('ALL_TENANTS', default=None), help='Shows details for all tenants. Admin only.')
@utils.arg('--filters', action=AppendFilters, type=str, nargs='*', start_version='3.33', metavar='<key=value>', default=None, help="Filter key and value pairs. Please use 'cinder list-filters' to check enabled filters from server. Use 'key~=value' for inexact filtering if the key supports. Default=None.")
def do_group_list(cs, args):
    """Lists all groups."""
    search_opts = {'all_tenants': args.all_tenants}
    if AppendFilters.filters:
        search_opts.update(shell_utils.extract_filters(AppendFilters.filters))
    groups = cs.groups.list(search_opts=search_opts)
    columns = ['ID', 'Status', 'Name']
    shell_utils.print_list(groups, columns)
    with cs.groups.completion_cache('uuid', cinderclient.v3.groups.Group, mode='w'):
        for group in groups:
            cs.groups.write_to_completion_cache('uuid', group.id)
    with cs.groups.completion_cache('name', cinderclient.v3.groups.Group, mode='w'):
        for group in groups:
            if group.name is None:
                continue
            cs.groups.write_to_completion_cache('name', group.name)
    AppendFilters.filters = []