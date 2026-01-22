import psycopg2
from psycopg2 import extensions
import eventlet.hubs
def make_psycopg_green():
    """Configure Psycopg to be used with eventlet in non-blocking way."""
    if not hasattr(extensions, 'set_wait_callback'):
        raise ImportError('support for coroutines not available in this Psycopg version (%s)' % psycopg2.__version__)
    extensions.set_wait_callback(eventlet_wait_callback)