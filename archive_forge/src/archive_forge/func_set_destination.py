import sys
import pprint
from docutils import __version__, __version_details__, SettingsSpec
from docutils import frontend, io, utils, readers, writers
from docutils.frontend import OptionParser
from docutils.transforms import Transformer
from docutils.utils.error_reporting import ErrorOutput, ErrorString
import docutils.readers.doctree
def set_destination(self, destination=None, destination_path=None):
    if destination_path is None:
        destination_path = self.settings._destination
    else:
        self.settings._destination = destination_path
    self.destination = self.destination_class(destination=destination, destination_path=destination_path, encoding=self.settings.output_encoding, error_handler=self.settings.output_encoding_error_handler)