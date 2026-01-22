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
def do_rate_limits(cs, args):
    """Lists rate limits for a user."""
    limits = cs.limits.get(args.tenant).rate
    columns = ['Verb', 'URI', 'Value', 'Remain', 'Unit', 'Next_Available']
    shell_utils.print_list(limits, columns)