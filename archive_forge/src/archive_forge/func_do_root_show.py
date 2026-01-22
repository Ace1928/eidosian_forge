import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance_or_cluster', metavar='<instance_or_cluster>', help=_('ID or name of the instance or cluster.'))
@utils.service_type('database')
def do_root_show(cs, args):
    """Gets status if root was ever enabled for an instance or cluster."""
    instance_or_cluster, resource_type = _find_instance_or_cluster(cs, args.instance_or_cluster)
    if resource_type == 'instance':
        root = cs.root.is_instance_root_enabled(instance_or_cluster)
    else:
        root = cs.root.is_cluster_root_enabled(instance_or_cluster)
    utils.print_dict({'is_root_enabled': root.rootEnabled})