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
@utils.arg('--detail', action='store_true', help='Show detailed information about pools.')
def do_get_pools(cs, args):
    """Show pool information for backends. Admin only."""
    pools = cs.volumes.get_pools(args.detail)
    infos = dict()
    infos.update(pools._info)
    for info in infos['pools']:
        backend = dict()
        backend['name'] = info['name']
        if args.detail:
            backend.update(info['capabilities'])
        shell_utils.print_dict(backend)