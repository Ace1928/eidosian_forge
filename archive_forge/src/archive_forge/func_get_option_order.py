import sys, string, re
import getopt
from distutils.errors import *
def get_option_order(self):
    """Returns the list of (option, value) tuples processed by the
        previous run of 'getopt()'.  Raises RuntimeError if
        'getopt()' hasn't been called yet.
        """
    if self.option_order is None:
        raise RuntimeError("'getopt()' hasn't been called yet")
    else:
        return self.option_order