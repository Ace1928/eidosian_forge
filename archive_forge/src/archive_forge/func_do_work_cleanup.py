import argparse
import collections
import os
from oslo_utils import strutils
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3.shell_base import *  # noqa
from cinderclient.v3.shell_base import CheckSizeArgForCreate
@api_versions.wraps('3.24')
@utils.arg('--cluster', metavar='<cluster-name>', default=None, help='Cluster name. Default=None.')
@utils.arg('--host', metavar='<hostname>', default=None, help='Service host name. Default=None.')
@utils.arg('--binary', metavar='<binary>', default=None, help='Service binary. Default=None.')
@utils.arg('--is-up', metavar='<True|true|False|false>', dest='is_up', default=None, choices=('True', 'true', 'False', 'false'), help='Filter by up/down status, if set to true services need to be up, if set to false services need to be down.  Default is None, which means up/down status is ignored.')
@utils.arg('--disabled', metavar='<True|true|False|false>', default=None, choices=('True', 'true', 'False', 'false'), help='Filter by disabled status. Default=None.')
@utils.arg('--resource-id', metavar='<resource-id>', default=None, help='UUID of a resource to cleanup. Default=None.')
@utils.arg('--resource-type', metavar='<Volume|Snapshot>', default=None, choices=('Volume', 'Snapshot'), help='Type of resource to cleanup.')
@utils.arg('--service-id', metavar='<service-id>', type=int, default=None, help='The service id field from the DB, not the uuid of the service. Default=None.')
def do_work_cleanup(cs, args):
    """Request cleanup of services with optional filtering."""
    filters = dict(cluster_name=args.cluster, host=args.host, binary=args.binary, is_up=args.is_up, disabled=args.disabled, resource_id=args.resource_id, resource_type=args.resource_type, service_id=args.service_id)
    filters = {k: v for k, v in filters.items() if v is not None}
    cleaning, unavailable = cs.workers.clean(**filters)
    columns = ('ID', 'Cluster Name', 'Host', 'Binary')
    if cleaning:
        print('Following services will be cleaned:')
        shell_utils.print_list(cleaning, columns)
    if unavailable:
        print('There are no alternative nodes to do cleanup for the following services:')
        shell_utils.print_list(unavailable, columns)
    if not (cleaning or unavailable):
        print('No cleanable services matched cleanup criteria.')