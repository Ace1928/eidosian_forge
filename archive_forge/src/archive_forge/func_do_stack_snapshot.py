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
@utils.arg('id', metavar='<NAME or ID>', help=_('Name or ID of stack to snapshot.'))
@utils.arg('-n', '--name', metavar='<NAME>', help=_('If specified, the name given to the snapshot.'))
def do_stack_snapshot(hc, args):
    """Make a snapshot of a stack."""
    show_deprecated('heat stack-snapshot', 'openstack stack snapshot create')
    fields = {'stack_id': args.id}
    if args.name:
        fields['name'] = args.name
    try:
        snapshot = hc.stacks.snapshot(**fields)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Stack not found: %s') % args.id)
    else:
        print(jsonutils.dumps(snapshot, indent=2, ensure_ascii=False))