from reportlab.pdfbase.pdfdoc import format, PDFObject, pdfdocEnc
from reportlab.lib.utils import strTypes

        Description of a kind of PDF object using a pattern.

        Pattern sequence should contain strings, singletons of form [string] or
        PDFPatternIf objects.
        Strings are literal strings to be used in the object.
        Singletons are names of keyword arguments to include.
        PDFpatternIf objects allow some conditionality.
        Keyword arguments can be non-instances which are substituted directly in string conversion,
        or they can be object instances in which case they should be pdfdoc.* style
        objects with a x.format(doc) method.
        Keyword arguments may be set on initialization or subsequently using __setitem__, before format.
        "constant object" instances can also be inserted in the patterns.
        