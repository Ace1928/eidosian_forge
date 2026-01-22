import yaml
from oslo_serialization import jsonutils
from zunclient.common import cliutils as utils
from zunclient.common import template_utils
from zunclient.common import utils as zun_utils
from zunclient.i18n import _
@utils.arg('-f', '--template-file', metavar='<file>', required=True, help=_('Path to the template.'))
def do_capsule_create(cs, args):
    """Create a capsule."""
    opts = {}
    if args.template_file:
        template = template_utils.get_template_contents(args.template_file)
        opts['template'] = template
        cs.capsules.create(**opts)
        print('Request to create capsule has been accepted.')