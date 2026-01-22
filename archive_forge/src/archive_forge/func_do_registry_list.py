import yaml
from oslo_serialization import jsonutils
from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
from zunclient import exceptions as exc
@utils.arg('--all-projects', action='store_true', default=False, help='List registries in all projects')
@utils.arg('--marker', metavar='<marker>', default=None, help='The last registry UUID of the previous page; displays list of registries after "marker".')
@utils.arg('--limit', metavar='<limit>', type=int, help='Maximum number of registries to return')
@utils.arg('--sort-key', metavar='<sort-key>', help='Column to sort results by')
@utils.arg('--sort-dir', metavar='<sort-dir>', choices=['desc', 'asc'], help='Direction to sort. "asc" or "desc".')
@utils.arg('--name', metavar='<name>', help='List registries according to their name.')
@utils.arg('--domain', metavar='<domain>', help='List registries according to their domain.')
@utils.arg('--project-id', metavar='<project-id>', help='List registries according to their Project_id')
@utils.arg('--user-id', metavar='<user-id>', help='List registries according to their user_id')
@utils.arg('--username', metavar='<username>', help='List registries according to their username')
def do_registry_list(cs, args):
    """Print a list of available registries."""
    opts = {}
    opts['all_projects'] = args.all_projects
    opts['marker'] = args.marker
    opts['limit'] = args.limit
    opts['sort_key'] = args.sort_key
    opts['sort_dir'] = args.sort_dir
    opts['domain'] = args.domain
    opts['name'] = args.name
    opts['project_id'] = args.project_id
    opts['user_id'] = args.user_id
    opts['username'] = args.username
    opts = zun_utils.remove_null_parms(**opts)
    registries = cs.registries.list(**opts)
    columns = ('uuid', 'name', 'domain', 'username', 'password')
    utils.print_list(registries, columns, sortby_index=None)