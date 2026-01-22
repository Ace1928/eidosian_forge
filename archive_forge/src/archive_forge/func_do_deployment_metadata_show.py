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
@utils.arg('id', metavar='<ID>', help=_('ID of the server to fetch deployments for.'))
def do_deployment_metadata_show(hc, args):
    """Get deployment configuration metadata for the specified server."""
    show_deprecated('heat deployment-metadata-show', 'openstack software deployment metadata show')
    md = hc.software_deployments.metadata(server_id=args.id)
    print(jsonutils.dumps(md, indent=2))