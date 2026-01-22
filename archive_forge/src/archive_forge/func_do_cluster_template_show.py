from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient.exceptions import InvalidAttribute
from magnumclient.i18n import _
from magnumclient.v1 import basemodels
@utils.arg('cluster_template', metavar='<cluster_template>', help=_('ID or name of the cluster template to show.'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_cluster_template_show(cs, args):
    """Show details about the given cluster template."""
    cluster_template = cs.cluster_templates.get(args.cluster_template)
    _show_cluster_template(cluster_template)