import inspect
from functools import partial
from weakref import WeakMethod
@classmethod
def register_event_type(cls, name):
    """Register an event type with the dispatcher.

        Registering event types allows the dispatcher to validate event
        handler names as they are attached, and to search attached objects for
        suitable handlers.

        :Parameters:
            `name` : str
                Name of the event to register.

        """
    if not hasattr(cls, 'event_types'):
        cls.event_types = []
    cls.event_types.append(name)
    return name