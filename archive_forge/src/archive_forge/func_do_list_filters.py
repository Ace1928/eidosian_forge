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
@api_versions.wraps('3.33')
@utils.arg('--resource', metavar='<resource>', default=None, help='Show enabled filters for specified resource. Default=None.')
def do_list_filters(cs, args):
    """List enabled filters.

    Symbol '~' after filter key means it supports inexact filtering.
    """
    filters = cs.resource_filters.list(resource=args.resource)
    shell_utils.print_resource_filter_list(filters)