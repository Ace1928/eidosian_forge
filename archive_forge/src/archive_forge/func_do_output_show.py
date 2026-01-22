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
@utils.arg('output', metavar='<OUTPUT NAME>', nargs='?', default=None, help=_('Name of an output to display.'))
@utils.arg('-F', '--format', metavar='<FORMAT>', help=_('The output value format, one of: json, raw.'), default='raw')
@utils.arg('-a', '--all', default=False, action='store_true', help=_('Display all stack outputs.'))
@utils.arg('--with-detail', default=False, action='store_true', help=_('Enable detail information presented, like key and description.'))
def do_output_show(hc, args):
    """Show a specific stack output."""
    show_deprecated('heat output-show', 'openstack stack output show')

    def resolve_output(output_key):
        try:
            output = hc.stacks.output_show(args.id, output_key)
        except exc.HTTPNotFound:
            try:
                output = None
                stack = hc.stacks.get(args.id).to_dict()
                for o in stack.get('outputs', []):
                    if o['output_key'] == output_key:
                        output = {'output': o}
                        break
                if output is None:
                    raise exc.CommandError(_('Output %(key)s not found.') % {'key': args.output})
            except exc.HTTPNotFound:
                raise exc.CommandError(_('Stack %(id)s or output %(key)s not found.') % {'id': args.id, 'key': args.output})
        return output

    def show_output(output):
        if 'output_error' in output['output']:
            msg = _('Output error: %s') % output['output']['output_error']
            raise exc.CommandError(msg)
        if args.with_detail:
            formatters = {'output_value': lambda x: utils.json_formatter(x) if args.format == 'json' else x}
            utils.print_dict(output['output'], formatters=formatters)
        elif args.format == 'json':
            print(utils.json_formatter(output['output']))
        elif isinstance(output['output']['output_value'], dict) or isinstance(output['output']['output_value'], list):
            print(utils.json_formatter(output['output']['output_value']))
        else:
            print(output['output']['output_value'])
    if args.all:
        if args.output:
            raise exc.CommandError(_("Can't specify an output name and the --all flag"))
        try:
            outputs = hc.stacks.output_list(args.id)
            resolved = False
        except exc.HTTPNotFound:
            try:
                outputs = hc.stacks.get(args.id).to_dict()
                resolved = True
            except exc.HTTPNotFound:
                raise exc.CommandError(_('Stack not found: %s') % args.id)
        for output in outputs['outputs']:
            if resolved:
                show_output({'output': output})
            else:
                show_output(resolve_output(output['output_key']))
    else:
        show_output(resolve_output(args.output))