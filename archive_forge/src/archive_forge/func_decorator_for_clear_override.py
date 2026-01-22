import sys
import fixtures
from functools import wraps
def decorator_for_clear_override(wrapped_function):

    @wraps(wrapped_function)
    def _wrapper(*args, **kwargs):
        group = 'oslo_messaging_notifications'
        if args[0] == 'notification_driver':
            args = ('driver', group)
        elif args[0] == 'notification_transport_url':
            args = ('transport_url', group)
        elif args[0] == 'notification_topics':
            args = ('topics', group)
        return wrapped_function(*args, **kwargs)
    _wrapper.wrapped = wrapped_function
    return _wrapper