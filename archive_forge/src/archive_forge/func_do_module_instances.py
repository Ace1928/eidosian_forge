import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('module', metavar='<module>', type=str, help=_('ID or name of the module.'))
@utils.arg('--include_clustered', action='store_true', default=False, help=_('Include instances that are part of a cluster (default %(default)s).'))
@utils.arg('--limit', metavar='<limit>', default=None, help=_('Return up to N number of the most recent results.'))
@utils.arg('--marker', metavar='<ID>', type=str, default=None, help=_('Begin displaying the results for IDs greater than the specified marker. When used with --limit, set this to the last ID displayed in the previous run.'))
@utils.service_type('database')
def do_module_instances(cs, args):
    """Lists the instances that have a particular module applied."""
    module = _find_module(cs, args.module)
    items = cs.modules.instances(module, limit=args.limit, marker=args.marker, include_clustered=args.include_clustered)
    instance_list = items
    while not args.limit and items.next:
        items = cs.modules.instances(module, marker=items.next)
        instance_list += items
    _print_instances(instance_list, utils.is_admin(cs))