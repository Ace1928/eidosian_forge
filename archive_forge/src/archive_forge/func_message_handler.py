import functools
from debugpy.common import json, log, messaging, util
@staticmethod
def message_handler(f):
    """Applied to a message handler to automatically lock and unlock the session
        for its duration, and to validate the session state.

        If the handler raises ComponentNotAvailable or JsonIOError, converts it to
        Message.cant_handle().
        """

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
    return lock_and_handle