from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
@utils.arg('host', metavar='<host>', help='Name of host.')
@utils.arg('binary', metavar='<binary>', help='Service binary.')
@utils.arg('--unset', dest='force_down', help='Unset the force state down of service.', action='store_false', default=True)
def do_service_force_down(cs, args):
    """Force Zun service to down or unset the force state."""
    res = cs.services.force_down(args.host, args.binary, args.force_down)
    utils.print_dict(res[1]['service'])