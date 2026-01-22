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
def filter_settings_spec(settings_spec, *exclude, **replace):
    """Return a copy of `settings_spec` excluding/replacing some settings.

    `settings_spec` is a tuple of configuration settings with a structure
    described for docutils.SettingsSpec.settings_spec.

    Optional positional arguments are names of to-be-excluded settings.
    Keyword arguments are option specification replacements.
    (See the html4strict writer for an example.)
    """
    settings = list(settings_spec)
    for i in range(2, len(settings), 3):
        newopts = []
        for opt_spec in settings[i]:
            opt_name = [opt_string[2:].replace('-', '_') for opt_string in opt_spec[1] if opt_string.startswith('--')][0]
            if opt_name in exclude:
                continue
            if opt_name in list(replace.keys()):
                newopts.append(replace[opt_name])
            else:
                newopts.append(opt_spec)
        settings[i] = tuple(newopts)
    return tuple(settings)