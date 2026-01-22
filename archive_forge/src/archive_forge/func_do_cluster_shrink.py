import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('cluster', metavar='<cluster>', help=_('ID or name of the cluster.'))
@utils.arg('instances', metavar='<instance>', nargs='+', default=[], help=_('Drop instance(s) from the cluster. Specify multiple ids to drop multiple instances.'))
@utils.service_type('database')
def do_cluster_shrink(cs, args):
    """Drops instances from a cluster."""
    cluster = _find_cluster(cs, args.cluster)
    instances = [{'id': _find_instance(cs, instance).id} for instance in args.instances]
    cs.clusters.shrink(cluster, instances=instances)