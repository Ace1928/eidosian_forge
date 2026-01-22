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
@utils.arg('id', metavar='<NAME or ID>', help=_('Name or ID of stack the resource belongs to.'))
@utils.arg('resource', metavar='<RESOURCE>', help=_('Name of the resource to signal.'))
@utils.arg('-D', '--data', metavar='<DATA>', help=_('JSON Data to send to the signal handler.'))
@utils.arg('-f', '--data-file', metavar='<FILE>', help=_('File containing JSON data to send to the signal handler.'))
def do_resource_signal(hc, args):
    """Send a signal to a resource."""
    show_deprecated('heat resource-signal', 'openstack stack resource signal')
    fields = {'stack_id': args.id, 'resource_name': args.resource}
    data = args.data
    data_file = args.data_file
    if data and data_file:
        raise exc.CommandError(_('Can only specify one of data and data-file'))
    if data_file:
        data_url = utils.normalise_file_path_to_url(data_file)
        data = request.urlopen(data_url).read()
    if data:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        try:
            data = jsonutils.loads(data)
        except ValueError as ex:
            raise exc.CommandError(_('Data should be in JSON format: %s') % ex)
        if not isinstance(data, dict):
            raise exc.CommandError(_('Data should be a JSON dict'))
        fields['data'] = data
    try:
        hc.resources.signal(**fields)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Stack or resource not found: %(id)s %(resource)s') % {'id': args.id, 'resource': args.resource})