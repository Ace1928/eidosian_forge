import sys
import time
from heatclient._i18n import _
from heatclient.common import utils
import heatclient.exc as exc
from heatclient.v1 import events as events_mod
def get_hook_events(hc, stack_id, event_args, nested_depth=0, hook_type='pre-create'):
    if hook_type == 'pre-create':
        stack_action_reason = 'Stack CREATE started'
        hook_event_reason = 'CREATE paused until Hook pre-create is cleared'
        hook_clear_event_reason = 'Hook pre-create is cleared'
    elif hook_type == 'pre-update':
        stack_action_reason = 'Stack UPDATE started'
        hook_event_reason = 'UPDATE paused until Hook pre-update is cleared'
        hook_clear_event_reason = 'Hook pre-update is cleared'
    elif hook_type == 'pre-delete':
        stack_action_reason = 'Stack DELETE started'
        hook_event_reason = 'DELETE paused until Hook pre-delete is cleared'
        hook_clear_event_reason = 'Hook pre-delete is cleared'
    else:
        raise exc.CommandError(_('Unexpected hook type %s') % hook_type)
    events = get_events(hc, stack_id=stack_id, event_args=event_args, nested_depth=nested_depth)
    stack_name = stack_id.split('/')[0]
    action_start_event = [e for e in enumerate(events) if e[1].resource_status_reason == stack_action_reason and e[1].stack_name == stack_name][-1]
    action_start_index = action_start_event[0]
    events = events[action_start_index:]
    resource_event_map = {}
    for e in events:
        stack_resource = (e.stack_name, e.resource_name)
        if e.resource_status_reason == hook_event_reason:
            resource_event_map[e.stack_name, e.resource_name] = e
        elif e.resource_status_reason == hook_clear_event_reason:
            if resource_event_map.get(stack_resource):
                del resource_event_map[e.stack_name, e.resource_name]
    return list(resource_event_map.values())