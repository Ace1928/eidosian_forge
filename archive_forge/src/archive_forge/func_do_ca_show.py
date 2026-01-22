import os.path
from magnumclient.common import cliutils as utils
from magnumclient.i18n import _
@utils.arg('postional_cluster', metavar='<cluster>', nargs='?', default=None, help=_('ID or name of the cluster.'))
@utils.arg('--cluster', metavar='<cluster>', default=None, help=_('ID or name of the cluster. %s') % utils.CLUSTER_DEPRECATION_HELP)
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_ca_show(cs, args):
    """Show details about the CA certificate for a cluster."""
    utils.validate_cluster_args(args.postional_cluster, args.cluster)
    args.cluster = args.postional_cluster or args.cluster
    opts = {'cluster_uuid': _get_target_uuid(cs, args)}
    cert = cs.certificates.get(**opts)
    _show_cert(cert)