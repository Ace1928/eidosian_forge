import time
from copy import copy
from kombu import Exchange
def group_from(type):
    """Get the group part of an event type name.

    Example:
        >>> group_from('task-sent')
        'task'

        >>> group_from('custom-my-event')
        'custom'
    """
    return type.split('-', 1)[0]