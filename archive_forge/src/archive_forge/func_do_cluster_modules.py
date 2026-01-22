import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('cluster', metavar='<cluster>', help=_('ID or name of the cluster.'))
@utils.service_type('database')
def do_cluster_modules(cs, args):
    """Lists all modules for each instance of a cluster."""
    cluster = _find_cluster(cs, args.cluster)
    instances = cluster._info['instances']
    module_list = []
    for instance in instances:
        new_list = cs.instances.modules(instance['id'])
        for item in new_list:
            item.instance_id = instance['id']
            item.instance_name = instance['name']
        module_list += new_list
    utils.print_list(module_list, ['instance_name', 'name', 'type', 'md5', 'created', 'updated'], labels={'name': 'Module Name', 'type': 'Module Type'})