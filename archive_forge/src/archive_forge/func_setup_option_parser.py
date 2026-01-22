import sys
import pprint
from docutils import __version__, __version_details__, SettingsSpec
from docutils import frontend, io, utils, readers, writers
from docutils.frontend import OptionParser
from docutils.transforms import Transformer
from docutils.utils.error_reporting import ErrorOutput, ErrorString
import docutils.readers.doctree
def setup_option_parser(self, usage=None, description=None, settings_spec=None, config_section=None, **defaults):
    if config_section:
        if not settings_spec:
            settings_spec = SettingsSpec()
        settings_spec.config_section = config_section
        parts = config_section.split()
        if len(parts) > 1 and parts[-1] == 'application':
            settings_spec.config_section_dependencies = ['applications']
    option_parser = OptionParser(components=(self.parser, self.reader, self.writer, settings_spec), defaults=defaults, read_config_files=True, usage=usage, description=description)
    return option_parser