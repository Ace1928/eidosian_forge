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
@utils.arg('tenant', metavar='<tenant_id>', nargs='?', default=None, help='Display information for a single tenant (Admin only).')
def do_absolute_limits(cs, args):
    """Lists absolute limits for a user."""
    limits = cs.limits.get(args.tenant).absolute
    columns = ['Name', 'Value']
    shell_utils.print_list(limits, columns)