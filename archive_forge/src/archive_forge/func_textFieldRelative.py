from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
def textFieldRelative(canvas, title, xR, yR, width, height, value='', maxlen=1000000, multiline=0):
    """same as textFieldAbsolute except the x and y are relative to the canvas coordinate transform"""
    xA, yA = canvas.absolutePosition(xR, yR)
    return textFieldAbsolute(canvas, title, xA, yA, width, height, value, maxlen, multiline)