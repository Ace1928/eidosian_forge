import sys
import pprint
from docutils import __version__, __version_details__, SettingsSpec
from docutils import frontend, io, utils, readers, writers
from docutils.frontend import OptionParser
from docutils.transforms import Transformer
from docutils.utils.error_reporting import ErrorOutput, ErrorString
import docutils.readers.doctree
def set_components(self, reader_name, parser_name, writer_name):
    if self.reader is None:
        self.set_reader(reader_name, self.parser, parser_name)
    if self.parser is None:
        if self.reader.parser is None:
            self.reader.set_parser(parser_name)
        self.parser = self.reader.parser
    if self.writer is None:
        self.set_writer(writer_name)