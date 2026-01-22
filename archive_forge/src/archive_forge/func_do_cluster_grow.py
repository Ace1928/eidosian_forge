import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('--' + INSTANCE_ARG_NAME, metavar=INSTANCE_METAVAR, action='append', dest='instances', default=[], help=INSTANCE_HELP)
@utils.arg('cluster', metavar='<cluster>', help=_('ID or name of the cluster.'))
@utils.service_type('database')
def do_cluster_grow(cs, args):
    """Adds more instances to a cluster."""
    cluster = _find_cluster(cs, args.cluster)
    instances = _parse_instance_options(cs, args.instances, for_grow=True)
    cs.clusters.grow(cluster, instances=instances)