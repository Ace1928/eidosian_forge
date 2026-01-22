import yaml
from oslo_serialization import jsonutils
from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
@utils.arg('host', metavar='<host>', help='ID or name of the host to show.')
@utils.arg('-f', '--format', metavar='<format>', action='store', choices=['json', 'yaml', 'table'], default='table', help='Print representation of the host.The choices of the output format is json,table,yaml.Defaults to table.')
def do_host_show(cs, args):
    """Show details of a host."""
    host = cs.hosts.get(args.host)
    if args.format == 'json':
        print(jsonutils.dumps(host._info, indent=4, sort_keys=True))
    elif args.format == 'yaml':
        print(yaml.safe_dump(host._info, default_flow_style=False))
    elif args.format == 'table':
        utils.print_dict(host._info)