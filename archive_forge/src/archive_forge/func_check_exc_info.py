import re
import sys
import warnings
def check_exc_info(exc_info):
    assert_(exc_info is None or type(exc_info) is tuple, 'exc_info (%r) is not a tuple: %r' % (exc_info, type(exc_info)))