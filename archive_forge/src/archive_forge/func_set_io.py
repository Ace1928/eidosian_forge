import sys
import pprint
from docutils import __version__, __version_details__, SettingsSpec
from docutils import frontend, io, utils, readers, writers
from docutils.frontend import OptionParser
from docutils.transforms import Transformer
from docutils.utils.error_reporting import ErrorOutput, ErrorString
import docutils.readers.doctree
def set_io(self, source_path=None, destination_path=None):
    if self.source is None:
        self.set_source(source_path=source_path)
    if self.destination is None:
        self.set_destination(destination_path=destination_path)