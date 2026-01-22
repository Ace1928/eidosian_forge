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
@utils.arg('resource_type', metavar='<RESOURCE_TYPE>', help=_('Resource type to get the details for.'))
def do_resource_type_show(hc, args):
    """Show the resource type."""
    show_deprecated('heat resource-type-show', 'openstack orchestration resource type show')
    try:
        resource_type = hc.resource_types.get(args.resource_type)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Resource Type not found: %s') % args.resource_type)
    else:
        print(jsonutils.dumps(resource_type, indent=2))