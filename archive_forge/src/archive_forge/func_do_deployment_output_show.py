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
@utils.arg('id', metavar='<ID>', help=_('ID deployment to show the output for.'))
@utils.arg('output', metavar='<OUTPUT NAME>', nargs='?', default=None, help=_('Name of an output to display.'))
@utils.arg('-a', '--all', default=False, action='store_true', help=_('Display all deployment outputs.'))
@utils.arg('-F', '--format', metavar='<FORMAT>', help=_('The output value format, one of: raw, json'), default='raw')
def do_deployment_output_show(hc, args):
    """Show a specific deployment output."""
    show_deprecated('heat deployment-output-show', 'openstack software deployment output show')
    if not args.all and args.output is None or (args.all and args.output is not None):
        raise exc.CommandError(_('Error: either %(output)s or %(all)s argument is needed.') % {'output': '<OUTPUT NAME>', 'all': '--all'})
    try:
        sd = hc.software_deployments.get(deployment_id=args.id)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Deployment not found: %s') % args.id)
    outputs = sd.to_dict().get('output_values', {})
    if args.all:
        print(utils.json_formatter(outputs))
    else:
        for output_key, value in outputs.items():
            if output_key == args.output:
                break
        else:
            return
        if args.format == 'json' or isinstance(value, dict) or isinstance(value, list):
            print(utils.json_formatter(value))
        else:
            print(value)