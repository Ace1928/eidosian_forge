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
def validate_smartquotes_locales(setting, value, option_parser, config_parser=None, config_section=None):
    """Check/normalize a comma separated list of smart quote definitions.

    Return a list of (language-tag, quotes) string tuples."""
    value = validate_comma_separated_list(setting, value, option_parser, config_parser, config_section)
    lc_quotes = []
    for item in value:
        try:
            lang, quotes = item.split(':', 1)
        except AttributeError:
            lc_quotes.append(item)
            continue
        except ValueError:
            raise ValueError('Invalid value "%s". Format is "<language>:<quotes>".' % item.encode('ascii', 'backslashreplace'))
        quotes = quotes.strip()
        multichar_quotes = quotes.split(':')
        if len(multichar_quotes) == 4:
            quotes = multichar_quotes
        elif len(quotes) != 4:
            raise ValueError('Invalid value "%s". Please specify 4 quotes\n    (primary open/close; secondary open/close).' % item.encode('ascii', 'backslashreplace'))
        lc_quotes.append((lang, quotes))
    return lc_quotes