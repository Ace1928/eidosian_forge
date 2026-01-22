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
@utils.arg('id', metavar='<NAME or ID>', nargs='+', help=_('Name or ID of stack(s) to delete.'))
@utils.arg('-y', '--yes', default=False, action='store_true', help=_('Skip yes/no prompt (assume yes).'))
def do_stack_delete(hc, args):
    """Delete the stack(s)."""
    show_deprecated('heat stack-delete', 'openstack stack delete')
    failure_count = 0
    try:
        if not args.yes and sys.stdin.isatty():
            prompt_response = input(_('Are you sure you want to delete this stack(s) [y/N]? ')).lower()
            if not prompt_response.startswith('y'):
                logger.info('User did not confirm stack delete so taking no action.')
                return
    except KeyboardInterrupt:
        logger.info('User did not confirm stack delete (ctrl-c) so taking no action.')
        return
    except EOFError:
        logger.info('User did not confirm stack delete (ctrl-d) so taking no action.')
        return
    for sid in args.id:
        fields = {'stack_id': sid}
        try:
            hc.stacks.delete(**fields)
            success_msg = _('Request to delete stack %s has been accepted.')
            print(success_msg % sid)
        except (exc.HTTPNotFound, exc.Forbidden) as e:
            failure_count += 1
            print(e)
    if failure_count:
        raise exc.CommandError(_('Unable to delete %(count)d of the %(total)d stacks.') % {'count': failure_count, 'total': len(args.id)})