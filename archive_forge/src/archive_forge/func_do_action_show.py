from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
@utils.arg('container', metavar='<container>', help='ID or name of the container whose actions are showed.')
@utils.arg('request_id', metavar='<request_id>', help='request ID of action to describe.')
def do_action_show(cs, args):
    """Describe a specific action."""
    action = cs.actions.get(args.container, args.request_id)
    _show_action(action)