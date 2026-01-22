import os
import os.path
import sys
import warnings
import configparser as CP
import codecs
import optparse
from optparse import SUPPRESS_HELP
import docutils
import docutils.utils
import docutils.nodes
from docutils.utils.error_reporting import (locale_encoding, SafeString,
def validate_encoding_error_handler(setting, value, option_parser, config_parser=None, config_section=None):
    try:
        codecs.lookup_error(value)
    except LookupError:
        raise LookupError('unknown encoding error handler: "%s" (choices: "strict", "ignore", "replace", "backslashreplace", "xmlcharrefreplace", and possibly others; see documentation for the Python ``codecs`` module)' % value)
    return value