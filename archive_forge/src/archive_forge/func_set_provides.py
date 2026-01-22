import sys
import os
import re
from email import message_from_file
from distutils.errors import *
from distutils.fancy_getopt import FancyGetopt, translate_longopt
from distutils.util import check_environ, strtobool, rfc822_escape
from distutils import log
from distutils.debug import DEBUG
def set_provides(self, value):
    value = [v.strip() for v in value]
    for v in value:
        import distutils.versionpredicate
        distutils.versionpredicate.split_provision(v)
    self.provides = value