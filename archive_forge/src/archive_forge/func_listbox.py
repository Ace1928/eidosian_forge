from reportlab.pdfbase.pdfdoc import (PDFObject, PDFArray, PDFDictionary, PDFString, pdfdocEnc,
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import Color, CMYKColor, Whiter, Blacker, opaqueColor
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import isStr, asNative
import weakref
def listbox(self, value='', fillColor=None, borderColor=None, textColor=None, borderWidth=1, borderStyle='solid', width=120, height=36, x=0, y=0, tooltip=None, name=None, annotationFlags='print', fieldFlags='', forceBorder=False, relative=False, fontName=None, fontSize=None, dashLen=3, maxlen=None, options=[]):
    return self._textfield(value=value, fillColor=fillColor, borderColor=borderColor, textColor=textColor, borderWidth=borderWidth, borderStyle=borderStyle, width=width, height=height, x=x, y=y, tooltip=tooltip, name=name, annotationFlags=annotationFlags, fieldFlags=fieldFlags, forceBorder=forceBorder, relative=relative, maxlen=maxlen, fontName=fontName, fontSize=fontSize, dashLen=dashLen, wkind='listbox', options=options)