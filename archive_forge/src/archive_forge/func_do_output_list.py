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
@utils.arg('id', metavar='<NAME or ID>', help=_('Name or ID of stack to query.'))
def do_output_list(hc, args):
    """Show available outputs."""
    show_deprecated('heat output-list', 'openstack stack output list')
    try:
        outputs = hc.stacks.output_list(args.id)
    except exc.HTTPNotFound:
        try:
            outputs = hc.stacks.get(args.id).to_dict()
        except exc.HTTPNotFound:
            raise exc.CommandError(_('Stack not found: %s') % args.id)
    fields = ['output_key', 'description']
    formatters = {'output_key': lambda x: x['output_key'], 'description': lambda x: x['description']}
    utils.print_list(outputs['outputs'], fields, formatters=formatters)