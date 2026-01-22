import sys
import pprint
from docutils import __version__, __version_details__, SettingsSpec
from docutils import frontend, io, utils, readers, writers
from docutils.frontend import OptionParser
from docutils.transforms import Transformer
from docutils.utils.error_reporting import ErrorOutput, ErrorString
import docutils.readers.doctree
def report_SystemMessage(self, error):
    print('Exiting due to level-%s (%s) system message.' % (error.level, utils.Reporter.levels[error.level]), file=self._stderr)