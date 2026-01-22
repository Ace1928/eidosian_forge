import inspect
from functools import partial
from weakref import WeakMethod
def remove_handlers(self, *args, **kwargs):
    """Remove event handlers from the event stack.

        See :py:meth:`~pyglet.event.EventDispatcher.push_handlers` for the
        accepted argument types. All handlers are removed from the first stack
        frame that contains any of the given handlers. No error is raised if
        any handler does not appear in that frame, or if no stack frame
        contains any of the given handlers.

        If the stack frame is empty after removing the handlers, it is
        removed from the stack.  Note that this interferes with the expected
        symmetry of :py:meth:`~pyglet.event.EventDispatcher.push_handlers` and
        :py:meth:`~pyglet.event.EventDispatcher.pop_handlers`.
        """
    handlers = list(self._get_handlers(args, kwargs))

    def find_frame():
        for frame in self._event_stack:
            for name, handler in handlers:
                try:
                    if frame[name] == handler:
                        return frame
                except KeyError:
                    pass
    frame = find_frame()
    if not frame:
        return
    for name, handler in handlers:
        try:
            if frame[name] == handler:
                del frame[name]
        except KeyError:
            pass
    if not frame:
        self._event_stack.remove(frame)