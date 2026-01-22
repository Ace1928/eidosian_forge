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
@utils.arg('--all-tenants', dest='all_tenants', metavar='<0|1>', nargs='?', type=int, const=1, default=0, help='Shows details for all tenants. Admin only.')
@utils.arg('--all_tenants', nargs='?', type=int, const=1, help=argparse.SUPPRESS)
def do_transfer_list(cs, args):
    """Lists all transfers."""
    all_tenants = int(os.environ.get('ALL_TENANTS', args.all_tenants))
    search_opts = {'all_tenants': all_tenants}
    transfers = cs.transfers.list(search_opts=search_opts)
    columns = ['ID', 'Volume ID', 'Name']
    shell_utils.print_list(transfers, columns)