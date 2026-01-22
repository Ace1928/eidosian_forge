import functools
from debugpy.common import json, log, messaging, util
@functools.wraps(f)
def lock_and_handle(self, message):
    try:
        with self.session:
            return f(self, message)
    except ComponentNotAvailable as exc:
        raise message.cant_handle('{0}', exc, silent=True)
    except messaging.MessageHandlingError as exc:
        if exc.cause is message:
            raise
        else:
            exc.propagate(message)
    except messaging.JsonIOError as exc:
        raise message.cant_handle('{0} disconnected unexpectedly', exc.stream.name, silent=True)