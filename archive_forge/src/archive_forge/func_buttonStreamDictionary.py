from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
def buttonStreamDictionary(width=16.7704, height=14.907):
    """everything except the length for the button appearance streams"""
    result = PDFDictionary()
    result['SubType'] = '/Form'
    result['BBox'] = '[0 0 %(width)s %(height)s]' % vars()
    font = PDFDictionary()
    font['ZaDb'] = PDFPattern(ZaDbPattern)
    resources = PDFDictionary()
    resources['ProcSet'] = '[ /PDF /Text ]'
    resources['Font'] = font
    result['Resources'] = resources
    return result