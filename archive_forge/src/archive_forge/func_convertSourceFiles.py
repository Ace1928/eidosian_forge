import re
from io import StringIO
def convertSourceFiles(filenames):
    """Helper function - makes minimal PDF document"""
    from reportlab.platypus import Paragraph, SimpleDocTemplate, XPreformatted
    from reportlab.lib.styles import getSampleStyleSheet
    styT = getSampleStyleSheet()['Title']
    styC = getSampleStyleSheet()['Code']
    doc = SimpleDocTemplate('pygments2xpre.pdf')
    S = [].append
    for filename in filenames:
        S(Paragraph(filename, style=styT))
        src = open(filename, 'r').read()
        fmt = pygments2xpre(src)
        S(XPreformatted(fmt, style=styC))
    doc.build(S.__self__)
    print('saved pygments2xpre.pdf')