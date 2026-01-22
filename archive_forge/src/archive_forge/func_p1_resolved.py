from promise import Promise
from .utils import assert_exception
from threading import Event
def p1_resolved(v):
    return pending