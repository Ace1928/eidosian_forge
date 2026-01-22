import logging
import sys
from oslo_serialization import jsonutils
from oslo_utils import strutils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import deployment_utils
from heatclient.common import event_utils
from heatclient.common import hook_utils
from heatclient.common import http
from heatclient.common import template_format
from heatclient.common import template_utils
from heatclient.common import utils
import heatclient.exc as exc
@utils.arg('-s', '--server', metavar='<SERVER>', help=_('ID of the server to fetch deployments for.'))
def do_deployment_list(hc, args):
    """List software deployments."""
    show_deprecated('heat deployment-list', 'openstack software deployment list')
    kwargs = {'server_id': args.server} if args.server else {}
    deployments = hc.software_deployments.list(**kwargs)
    fields = ['id', 'config_id', 'server_id', 'action', 'status', 'creation_time', 'status_reason']
    utils.print_list(deployments, fields, sortby_index=5)