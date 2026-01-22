from reportlab.pdfbase.pdfdoc import (PDFObject, PDFArray, PDFDictionary, PDFString, pdfdocEnc,
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import Color, CMYKColor, Whiter, Blacker, opaqueColor
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import isStr, asNative
import weakref
def makeFont(self, fontName):
    if fontName is None:
        fontName = 'Helvetica'
    if fontName not in self.formFontNames:
        raise ValueError('form font name, %r, is not one of the standard 14 fonts' % fontName)
    fn = self.formFontNames[fontName]
    ref = self.getRefStr(PDFFromString('<< /BaseFont /%s /Subtype /Type1 /Name /%s /Type /Font /Encoding %s >>' % (fontName, fn, self.encRefStr)))
    if fn not in self.fonts:
        self.fonts[fn] = ref
    return (ref, fn)