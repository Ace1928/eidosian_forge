import os.path
from magnumclient.common import cliutils as utils
from magnumclient.i18n import _
@utils.arg('--cluster', required=True, metavar='<cluster>', help=_('ID or name of the cluster.'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_ca_rotate(cs, args):
    """Rotate the CA certificate for a cluster to revoke access."""
    cluster = cs.clusters.get(args.cluster)
    opts = {'cluster_uuid': cluster.uuid}
    cs.certificates.rotate_ca(**opts)