from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
def textFieldAbsolute(canvas, title, x, y, width, height, value='', maxlen=1000000, multiline=0):
    """Place a text field on the current page
        with name title at ABSOLUTE position (x,y) with
        dimensions (width, height), using value as the default value and
        maxlen as the maximum permissible length.  If multiline is set make
        it a multiline field.
    """
    theform = getForm(canvas)
    return theform.textField(canvas, title, x, y, x + width, y + height, value, maxlen, multiline)