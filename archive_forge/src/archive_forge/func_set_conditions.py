import sys
import os
import os.path
import re
import itertools
import warnings
import unicodedata
from docutils import ApplicationError, DataError, __version_info__
from docutils import nodes
from docutils.nodes import unescape
import docutils.io
from docutils.utils.error_reporting import ErrorOutput, SafeString
def set_conditions(self, category, report_level, halt_level, stream=None, debug=False):
    warnings.warn('docutils.utils.Reporter.set_conditions deprecated; set attributes via configuration settings or directly', DeprecationWarning, stacklevel=2)
    self.report_level = report_level
    self.halt_level = halt_level
    if not isinstance(stream, ErrorOutput):
        stream = ErrorOutput(stream, self.encoding, self.error_handler)
    self.stream = stream
    self.debug_flag = debug