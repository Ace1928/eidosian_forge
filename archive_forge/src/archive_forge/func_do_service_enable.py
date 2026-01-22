from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
@utils.arg('host', metavar='<host>', help='Name of host.')
@utils.arg('binary', metavar='<binary>', help='Service binary.')
def do_service_enable(cs, args):
    """Enable the Zun service."""
    res = cs.services.enable(args.host, args.binary)
    utils.print_dict(res[1]['service'])