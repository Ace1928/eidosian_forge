import locale
import sys
from PyQt5.QtCore import (PYQT_VERSION_STR, QDir, QFile, QFileInfo, QIODevice,
from .pylupdate import *
def printUsage():
    sys.stderr.write('Usage:\n    pylupdate5 [options] project-file\n    pylupdate5 [options] source-files -ts ts-files\n\nOptions:\n    -help  Display this information and exit\n    -version\n           Display the version of pylupdate5 and exit\n    -verbose\n           Explain what is being done\n    -noobsolete\n           Drop all obsolete strings\n    -tr-function name\n           name() may be used instead of tr()\n    -translate-function name\n           name() may be used instead of translate()\n')