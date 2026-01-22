from celery import bootsteps
from celery.utils.log import get_logger
from .events import Events
def on_node_reply(self, c, nodename, reply):
    debug('mingle: processing reply from %s', nodename)
    try:
        self.sync_with_node(c, **reply)
    except MemoryError:
        raise
    except Exception as exc:
        exception('mingle: sync with %s failed: %r', nodename, exc)