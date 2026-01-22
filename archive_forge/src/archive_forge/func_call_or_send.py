import collections
import functools
import inspect
from oslo_log import log as logging
from oslo_messaging import rpc
@functools.wraps(function)
def call_or_send(processor, *args, **kwargs):
    if len(args) == 1 and (not kwargs) and isinstance(args[0], MessageData):
        try:
            return function(processor, **args[0]._asdict())
        except rpc.dispatcher.ExpectedException as exc:
            LOG.error('[%s] Exception in "%s": %s', processor.name, function.__name__, exc.exc_info[1], exc_info=exc.exc_info)
            raise
        except Exception as exc:
            LOG.exception('[%s] Exception in "%s": %s', processor.name, function.__name__, exc)
            raise
    else:
        data = inspect.getcallargs(function, processor, *args, **kwargs)
        data.pop(arg_names[0])
        return processor.queue.send(function.__name__, MessageData(**data))