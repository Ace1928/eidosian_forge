import re
import hashlib
from string import digits
from math import sin, cos, tan, pi
from reportlab import rl_config
from reportlab.pdfbase import pdfdoc
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen  import pathobject
from reportlab.pdfgen.textobject import PDFTextObject, _PDFColorSetter
from reportlab.lib.colors import black, _chooseEnforceColorSpace, Color, CMYKColor, toColor
from reportlab.lib.utils import ImageReader, isSeq, isStr, isUnicode, _digester, asUnicode
from reportlab.lib.rl_accel import fp_str, escapePDF
from reportlab.lib.boxstuff import aspectRatioFix
def setOutlineNames0(self, *nametree):
    """nametree should can be a recursive tree like so::
            
               c.setOutlineNames(
                 "chapter1dest",
                 ("chapter2dest",
                  ["chapter2section1dest",
                   "chapter2section2dest",
                   "chapter2conclusiondest"]
                 ), # end of chapter2 description
                 "chapter3dest",
                 ("chapter4dest", ["c4s1", "c4s2"])
                 )
          
          each of the string names inside must be bound to a bookmark
          before the document is generated.
        """
    self._doc.outline.setNames(*(self,) + nametree)