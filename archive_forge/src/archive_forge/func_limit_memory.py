import contextlib
import errno
import os
import resource
import sys
from breezy import osutils, tests
from breezy.tests import features, script
@contextlib.contextmanager
def limit_memory(size):
    if sys.platform in ('win32', 'darwin'):
        raise NotImplementedError
    previous = resource.getrlimit(RESOURCE)
    resource.setrlimit(RESOURCE, (LIMIT, -1))
    yield
    resource.setrlimit(RESOURCE, previous)