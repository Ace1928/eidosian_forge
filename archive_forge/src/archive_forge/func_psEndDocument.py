import math
from io import StringIO
from rdkit.sping.pid import *
from . import psmetrics  # for font info
def psEndDocument(self):
    if self._inDocumentFlag:
        self.code.append('%%Trailer')
        self.code.append('%%%%Pages: %d' % self.pageNum)
        self.code.append('%%EOF')