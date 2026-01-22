import yaml
from oslo_serialization import jsonutils
from zunclient.common import cliutils as utils
from zunclient.common import template_utils
from zunclient.common import utils as zun_utils
from zunclient.i18n import _
@utils.arg('capsule', metavar='<capsule>', help='ID or name of the capsule to show.')
@utils.arg('-f', '--format', metavar='<format>', action='store', choices=['json', 'yaml', 'table'], default='table', help='Print representation of the capsule. The choices of the output format is json,table,yaml. Defaults to table. ')
def do_capsule_describe(cs, args):
    """Show details of a capsule."""
    capsule = cs.capsules.describe(args.capsule)
    if args.format == 'json':
        print(jsonutils.dumps(capsule._info, indent=4, sort_keys=True))
    elif args.format == 'yaml':
        print(yaml.safe_dump(capsule._info, default_flow_style=False))
    elif args.format == 'table':
        _show_capsule(capsule)