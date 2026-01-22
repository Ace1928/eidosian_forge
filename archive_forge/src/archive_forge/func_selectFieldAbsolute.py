from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
def selectFieldAbsolute(canvas, title, value, options, x, y, width, height):
    """Place a select field (drop down list) on the current page
        with name title and
        with options listed in the sequence options
        default value value (must be one of options)
        at ABSOLUTE position (x,y) with dimensions (width, height)."""
    theform = getForm(canvas)
    theform.selectField(canvas, title, value, options, x, y, x + width, y + height)