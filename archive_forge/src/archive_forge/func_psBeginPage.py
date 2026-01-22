import math
from io import StringIO
from rdkit.sping.pid import *
from . import psmetrics  # for font info
def psBeginPage(self, pageName=None):
    if not pageName:
        pageName = '%s %s' % (self.pageNum, self.pageNum)
    pagesetup = self._psPageSetupStr(self.size[1], self.defaultLineColor, self._findFont(self.defaultFont), self.defaultFont.size, self.defaultLineWidth)
    self.code.append(self.dsc.BeginPageStr(pageSetupStr=pagesetup, pageName=pageName))
    self._inPageFlag = 1