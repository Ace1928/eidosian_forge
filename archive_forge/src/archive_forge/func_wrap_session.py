import contextlib
import logging as log
from oslo_utils import reflection
from osprofiler import profiler
@contextlib.contextmanager
def wrap_session(sqlalchemy, sess):
    with sess as s:
        if not getattr(s.bind, 'traced', False):
            add_tracing(sqlalchemy, s.bind, 'db')
            s.bind.traced = True
        yield s