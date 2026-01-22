import sys
import pprint
from docutils import __version__, __version_details__, SettingsSpec
from docutils import frontend, io, utils, readers, writers
from docutils.frontend import OptionParser
from docutils.transforms import Transformer
from docutils.utils.error_reporting import ErrorOutput, ErrorString
import docutils.readers.doctree
def report_UnicodeError(self, error):
    data = error.object[error.start:error.end]
    self._stderr.write('%s\n\nThe specified output encoding (%s) cannot\nhandle all of the output.\nTry setting "--output-encoding-error-handler" to\n\n* "xmlcharrefreplace" (for HTML & XML output);\n  the output will contain "%s" and should be usable.\n* "backslashreplace" (for other output formats);\n  look for "%s" in the output.\n* "replace"; look for "?" in the output.\n\n"--output-encoding-error-handler" is currently set to "%s".\n\nExiting due to error.  Use "--traceback" to diagnose.\nIf the advice above doesn\'t eliminate the error,\nplease report it to <docutils-users@lists.sf.net>.\nInclude "--traceback" output, Docutils version (%s),\nPython version (%s), your OS type & version, and the\ncommand line used.\n' % (ErrorString(error), self.settings.output_encoding, data.encode('ascii', 'xmlcharrefreplace'), data.encode('ascii', 'backslashreplace'), self.settings.output_encoding_error_handler, __version__, sys.version.split()[0]))