import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def tokenize_annotated(doc, annotation):
    """Tokenize a document and add an annotation attribute to each token
    """
    tokens = tokenize(doc, include_hrefs=False)
    for tok in tokens:
        tok.annotation = annotation
    return tokens