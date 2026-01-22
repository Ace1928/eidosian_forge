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
@api_versions.wraps('3.11')
@utils.arg('--filters', action=AppendFilters, type=str, nargs='*', start_version='3.52', metavar='<key=value>', default=None, help='Filter key and value pairs. Admin only.')
def do_group_type_list(cs, args):
    """Lists available 'group types'. (Admin only will see private types)"""
    search_opts = {}
    if AppendFilters.filters:
        search_opts.update(shell_utils.extract_filters(AppendFilters.filters))
    gtypes = cs.group_types.list(search_opts=search_opts)
    shell_utils.print_group_type_list(gtypes)
    AppendFilters.filters = []