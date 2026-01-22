import sys
import time
from heatclient._i18n import _
from heatclient.common import utils
import heatclient.exc as exc
from heatclient.v1 import events as events_mod
def wait_for_events(ws, stack_name, out=None):
    """Receive events over the passed websocket and wait for final status."""
    msg_template = _('\n Stack %(name)s %(status)s \n')
    if not out:
        out = sys.stdout
    event_log_context = utils.EventLogContext()
    while True:
        data = ws.recv()['body']
        event = events_mod.Event(None, data['payload'], True)
        event.event_time = data['timestamp']
        event.resource_status = '%s_%s' % (event.resource_action, event.resource_status)
        events_log = utils.event_log_formatter([event], event_log_context)
        out.write(events_log)
        out.write('\n')
        if data['payload']['resource_name'] == stack_name:
            stack_status = data['payload']['resource_status']
            if stack_status in ('COMPLETE', 'FAILED'):
                msg = msg_template % dict(name=stack_name, status=event.resource_status)
                return ('%s_%s' % (event.resource_action, stack_status), msg)