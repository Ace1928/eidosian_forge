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
@utils.arg('-s', '--show-deleted', default=False, action='store_true', help=_('Include soft-deleted stacks in the stack listing.'))
@utils.arg('-n', '--show-nested', default=False, action='store_true', help=_('Include nested stacks in the stack listing.'))
@utils.arg('-a', '--show-hidden', default=False, action='store_true', help=_('Include hidden stacks in the stack listing.'))
@utils.arg('-f', '--filters', metavar='<KEY1=VALUE1;KEY2=VALUE2...>', help=_('Filter parameters to apply on returned stacks. This can be specified multiple times, or once with parameters separated by a semicolon.'), action='append')
@utils.arg('-t', '--tags', metavar='<TAG1,TAG2...>', help=_('Show stacks containing these tags. If multiple tags are passed, they will be combined using the AND boolean expression. '))
@utils.arg('--tags-any', metavar='<TAG1,TAG2...>', help=_('Show stacks containing these tags, If multiple tags are passed, they will be combined using the OR boolean expression. '))
@utils.arg('--not-tags', metavar='<TAG1,TAG2...>', help=_('Show stacks not containing these tags, If multiple tags are passed, they will be combined using the AND boolean expression. '))
@utils.arg('--not-tags-any', metavar='<TAG1,TAG2...>', help=_('Show stacks not containing these tags, If multiple tags are passed, they will be combined using the OR boolean expression. '))
@utils.arg('-l', '--limit', metavar='<LIMIT>', help=_('Limit the number of stacks returned.'))
@utils.arg('-m', '--marker', metavar='<ID>', help=_('Only return stacks that appear after the given stack ID.'))
@utils.arg('-k', '--sort-keys', metavar='<KEY1;KEY2...>', help=_('List of keys for sorting the returned stacks. This can be specified multiple times or once with keys separated by semicolons. Valid sorting keys include "stack_name", "stack_status", "creation_time" and "updated_time".'), action='append')
@utils.arg('-d', '--sort-dir', metavar='[asc|desc]', help=_('Sorting direction (either "asc" or "desc") for the sorting keys.'))
@utils.arg('-g', '--global-tenant', action='store_true', default=False, help=_('Display stacks from all tenants. Operation only authorized for users who match the policy (default or explicitly configured in policy.json) in heat.'))
@utils.arg('-o', '--show-owner', action='store_true', default=False, help=_('Display stack owner information. This is automatically enabled when using %(arg)s.') % {'arg': '--global-tenant'})
def do_stack_list(hc, args=None):
    """List the user's stacks."""
    show_deprecated('heat stack-list', 'openstack stack list')
    kwargs = {}
    fields = ['id', 'stack_name', 'stack_status', 'creation_time', 'updated_time']
    sort_keys = ['stack_name', 'stack_status', 'creation_time', 'updated_time']
    sortby_index = 3
    if args:
        kwargs = {'limit': args.limit, 'marker': args.marker, 'filters': utils.format_parameters(args.filters), 'tags': args.tags, 'tags_any': args.tags_any, 'not_tags': args.not_tags, 'not_tags_any': args.not_tags_any, 'global_tenant': args.global_tenant, 'show_deleted': args.show_deleted, 'show_hidden': args.show_hidden}
        if args.show_nested:
            fields.append('parent')
            kwargs['show_nested'] = True
        if args.sort_keys:
            keys = []
            for k in args.sort_keys:
                if ';' in k:
                    keys.extend(k.split(';'))
                else:
                    keys.append(k)
            for key in keys:
                if key not in sort_keys:
                    err = _("Sorting key '%(key)s' not one of the supported keys: %(keys)s") % {'key': key, 'keys': sort_keys}
                    raise exc.CommandError(err)
            kwargs['sort_keys'] = keys
            sortby_index = None
        if args.sort_dir:
            if args.sort_dir not in ('asc', 'desc'):
                raise exc.CommandError(_("Sorting direction must be one of 'asc' and 'desc'"))
            kwargs['sort_dir'] = args.sort_dir
        if args.global_tenant or args.show_owner:
            fields.append('stack_owner')
        if args.show_deleted:
            fields.append('deletion_time')
    stacks = hc.stacks.list(**kwargs)
    stacks = list(stacks)
    for stk in stacks:
        if hasattr(stk, 'project'):
            fields.append('project')
            break
    utils.print_list(stacks, fields, sortby_index=sortby_index)