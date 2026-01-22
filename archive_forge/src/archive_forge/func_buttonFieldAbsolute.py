from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
def buttonFieldAbsolute(canvas, title, value, x, y, width=16.7704, height=14.907):
    """Place a check button field on the current page
        with name title and default value value (one of "Yes" or "Off")
        at ABSOLUTE position (x,y).
    """
    theform = getForm(canvas)
    return theform.buttonField(canvas, title, value, x, y, width=width, height=height)