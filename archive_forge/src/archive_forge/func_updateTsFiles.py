import locale
import sys
from PyQt5.QtCore import (PYQT_VERSION_STR, QDir, QFile, QFileInfo, QIODevice,
from .pylupdate import *
def updateTsFiles(fetchedTor, tsFileNames, codecForTr, noObsolete, verbose):
    dir = QDir()
    for t in tsFileNames:
        fn = dir.relativeFilePath(t)
        tor = MetaTranslator()
        out = MetaTranslator()
        tor.load(t)
        if codecForTr:
            tor.setCodec(codecForTr)
        merge(tor, fetchedTor, out, noObsolete, verbose, fn)
        if noObsolete:
            out.stripObsoleteMessages()
        out.stripEmptyContexts()
        if not out.save(t):
            sys.stderr.write("pylupdate5 error: Cannot save '%s'\n" % fn)