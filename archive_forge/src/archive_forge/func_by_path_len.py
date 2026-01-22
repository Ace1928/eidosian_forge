import sys
from . import server
from .workers import threadpool
from ._compat import ntob, bton
def by_path_len(app):
    return len(app[0])