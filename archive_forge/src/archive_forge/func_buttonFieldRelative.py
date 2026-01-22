from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
def buttonFieldRelative(canvas, title, value, xR, yR, width=16.7704, height=14.907):
    """same as buttonFieldAbsolute except the x and y are relative to the canvas coordinate transform"""
    xA, yA = canvas.absolutePosition(xR, yR)
    return buttonFieldAbsolute(canvas, title, value, xA, yA, width=width, height=height)