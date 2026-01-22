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
def validate_ternary(setting, value, option_parser, config_parser=None, config_section=None):
    """Check/normalize three-value settings:
         True:  '1', 'on', 'yes', 'true'
         False: '0', 'off', 'no','false', ''
         any other value: returned as-is.
    """
    if isinstance(value, bool) or value is None:
        return value
    try:
        return option_parser.booleans[value.strip().lower()]
    except KeyError:
        return value