import os
from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient import exceptions
from magnumclient.i18n import _
@utils.arg('cluster', metavar='<cluster>', help=_('ID or name of the cluster to retrieve config.'))
@utils.arg('--dir', metavar='<dir>', default='.', help=_('Directory to save the certificate and config files.'))
@utils.arg('--force', action='store_true', default=False, help=_('Overwrite files if existing.'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_cluster_config(cs, args):
    """Configure native client to access cluster.

    You can source the output of this command to get the native client of the
    corresponding COE configured to access the cluster.

    Example: eval $(magnum cluster-config <cluster-name>).
    """
    args.dir = os.path.abspath(args.dir)
    cluster = cs.clusters.get(args.cluster)
    if hasattr(cluster, 'api_address') and cluster.api_address is None:
        print("WARNING: The cluster's api_address is not known yet.")
    cluster_template = cs.cluster_templates.get(cluster.cluster_template_id)
    opts = {'cluster_uuid': cluster.uuid}
    if not cluster_template.tls_disabled:
        tls = magnum_utils.generate_csr_and_key()
        tls['ca'] = cs.certificates.get(**opts).pem
        opts['csr'] = tls['csr']
        tls['cert'] = cs.certificates.create(**opts).pem
        for k in ('key', 'cert', 'ca'):
            fname = '%s/%s.pem' % (args.dir, k)
            if os.path.exists(fname) and (not args.force):
                raise Exception('File %s exists, aborting.' % fname)
            else:
                f = open(fname, 'w')
                f.write(tls[k])
                f.close()
    print(magnum_utils.config_cluster(cluster, cluster_template, cfg_dir=args.dir, force=args.force))