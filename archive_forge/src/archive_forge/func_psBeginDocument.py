import math
from io import StringIO
from rdkit.sping.pid import *
from . import psmetrics  # for font info
def psBeginDocument(self):
    self.code.append(self.dsc.documentHeader())
    self.code.append(self.dsc.boundingBoxStr(0, 0, self.size[0], self.size[1]))
    self.code.append('%%Pages: (atend)')
    self._inDocumentFlag = 1
    shapes = {'Helvetica': ['Roman', 'Bold', 'Oblique'], 'Times': ['Roman', 'Bold', 'Italic'], 'Courier': ['Roman', 'Bold', 'Oblique']}
    fntnames = []
    for basename in ['Helvetica', 'Times', 'Courier']:
        for mys in shapes[basename]:
            fntnames.append(basename + '-' + mys)
    for fontname in fntnames:
        self.code.append(latin1FontEncoding(fontname))
    self.code.append(dashLineDefinition())