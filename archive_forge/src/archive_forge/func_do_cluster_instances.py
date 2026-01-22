import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('cluster', metavar='<cluster>', help=_('ID or name of the cluster.'))
@utils.service_type('database')
def do_cluster_instances(cs, args):
    """Lists all instances of a cluster."""
    cluster = _find_cluster(cs, args.cluster)
    instances = cluster._info['instances']
    for instance in instances:
        instance['flavor_id'] = instance['flavor']['id']
        if instance.get('volume'):
            instance['size'] = instance['volume']['size']
    utils.print_list(instances, ['id', 'name', 'flavor_id', 'size', 'status'], obj_is_dict=True)