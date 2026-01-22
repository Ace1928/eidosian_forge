from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
@utils.arg('host', metavar='<host>', help='Name of host.')
@utils.arg('binary', metavar='<binary>', help='Service binary.')
@utils.arg('--reason', metavar='<reason>', help='Reason for disabling service.')
def do_service_disable(cs, args):
    """Disable the Zun service."""
    res = cs.services.disable(args.host, args.binary, args.reason)
    utils.print_dict(res[1]['service'])