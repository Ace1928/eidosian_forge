import threading
import sys
from paste.util import filemixin
def not_installed_error_stdin(*args, **kw):
    assert False, 'threadedprint has not yet been installed for stdin (call threadedprint.install_stdin())'