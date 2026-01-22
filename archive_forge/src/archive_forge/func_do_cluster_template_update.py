from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient.exceptions import InvalidAttribute
from magnumclient.i18n import _
from magnumclient.v1 import basemodels
@utils.arg('cluster_template', metavar='<cluster_template>', help=_('UUID or name of cluster template'))
@utils.arg('op', metavar='<op>', choices=['add', 'replace', 'remove'], help=_("Operations: 'add', 'replace' or 'remove'"))
@utils.arg('attributes', metavar='<path=value>', nargs='+', action='append', default=[], help=_('Attributes to add/replace or remove (only PATH is necessary on remove)'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_cluster_template_update(cs, args):
    """Updates one or more cluster template attributes."""
    patch = magnum_utils.args_array_to_patch(args.op, args.attributes[0])
    cluster_template = cs.cluster_templates.update(args.cluster_template, patch)
    _show_cluster_template(cluster_template)