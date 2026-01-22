from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
@utils.arg('container', metavar='<container>', help='ID or name of a container.')
def do_action_list(cs, args):
    """Print a list of actions done on a container."""
    container = args.container
    actions = cs.actions.list(container)
    columns = ('user_id', 'container_uuid', 'request_id', 'action', 'message', 'start_time')
    utils.print_list(actions, columns, {'versions': zun_utils.print_list_field('versions')}, sortby_index=None)