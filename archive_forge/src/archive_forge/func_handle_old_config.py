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
def handle_old_config(self, filename):
    warnings.warn_explicit(self.old_warning, ConfigDeprecationWarning, filename, 0)
    options = self.get_section('options')
    if not self.has_section('general'):
        self.add_section('general')
    for key, value in list(options.items()):
        if key in self.old_settings:
            section, setting = self.old_settings[key]
            if not self.has_section(section):
                self.add_section(section)
        else:
            section = 'general'
            setting = key
        if not self.has_option(section, setting):
            self.set(section, setting, value)
    self.remove_section('options')