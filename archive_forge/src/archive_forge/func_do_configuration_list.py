import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('--limit', metavar='<limit>', type=int, default=None, help=_('Limit the number of results displayed.'))
@utils.arg('--marker', metavar='<ID>', type=str, default=None, help=_('Begin displaying the results for IDs greater than the specified marker. When used with --limit, set this to the last ID displayed in the previous run.'))
@utils.service_type('database')
def do_configuration_list(cs, args):
    """Lists all configuration groups."""
    config_grps = cs.configurations.list(limit=args.limit, marker=args.marker)
    utils.print_list(config_grps, ['id', 'name', 'description', 'datastore_name', 'datastore_version_name'])