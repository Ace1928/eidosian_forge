import os
from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient import exceptions
from magnumclient.i18n import _
@utils.arg('--marker', metavar='<marker>', default=None, help=_('The last cluster UUID of the previous page; displays list of clusters after "marker".'))
@utils.arg('--limit', metavar='<limit>', type=int, help=_('Maximum number of clusters to return.'))
@utils.arg('--sort-key', metavar='<sort-key>', help=_('Column to sort results by.'))
@utils.arg('--sort-dir', metavar='<sort-dir>', choices=['desc', 'asc'], help=_('Direction to sort. "asc" or "desc".'))
@utils.arg('--fields', default=None, metavar='<fields>', help=_('Comma-separated list of fields to display. Available fields: uuid, name, cluster_template_id, stack_id, status, master_count, node_count, links, create_timeout'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_cluster_list(cs, args):
    """Print a list of available clusters."""
    clusters = cs.clusters.list(marker=args.marker, limit=args.limit, sort_key=args.sort_key, sort_dir=args.sort_dir)
    columns = ['uuid', 'name', 'keypair', 'node_count', 'master_count', 'status']
    columns += utils._get_list_table_columns_and_formatters(args.fields, clusters, exclude_fields=(c.lower() for c in columns))[0]
    utils.print_list(clusters, columns, {'versions': magnum_utils.print_list_field('versions')}, sortby_index=None)