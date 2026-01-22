import sys, string, re
import getopt
from distutils.errors import *
def get_attr_name(self, long_option):
    """Translate long option name 'long_option' to the form it
        has as an attribute of some object: ie., translate hyphens
        to underscores."""
    return long_option.translate(longopt_xlate)