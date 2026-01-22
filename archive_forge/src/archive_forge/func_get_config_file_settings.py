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
def get_config_file_settings(self, config_file):
    """Returns a dictionary containing appropriate config file settings."""
    parser = ConfigParser()
    parser.read(config_file, self)
    self.config_files.extend(parser._files)
    base_path = os.path.dirname(config_file)
    applied = {}
    settings = Values()
    for component in self.components:
        if not component:
            continue
        for section in tuple(component.config_section_dependencies or ()) + (component.config_section,):
            if section in applied:
                continue
            applied[section] = 1
            settings.update(parser.get_section(section), self)
    make_paths_absolute(settings.__dict__, self.relative_path_settings, base_path)
    return settings.__dict__