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
@utils.arg('-O', '--output-file', metavar='<FILE>', help=_('file to output abandon result. If the option is specified, the result will be output into <FILE>.'))
@utils.arg('id', metavar='<NAME or ID>', help=_('Name or ID of stack to abandon.'))
def do_stack_abandon(hc, args):
    """Abandon the stack.

    This will delete the record of the stack from Heat, but will not delete
    any of the underlying resources. Prints an adoptable JSON representation
    of the stack to stdout or a file on success.
    """
    show_deprecated('heat stack-abandon', 'openstack stack abandon')
    fields = {'stack_id': args.id}
    try:
        stack = hc.stacks.abandon(**fields)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Stack not found: %s') % args.id)
    else:
        result = jsonutils.dumps(stack, indent=2)
        if args.output_file is not None:
            try:
                with open(args.output_file, 'w') as f:
                    f.write(result)
            except IOError as err:
                print(result)
                raise exc.CommandError(str(err))
        else:
            print(result)