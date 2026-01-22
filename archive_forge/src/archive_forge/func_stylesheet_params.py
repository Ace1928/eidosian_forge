import sys
import os.path
from lxml import etree as _etree # due to validator __init__ signature
def stylesheet_params(**kwargs):
    """Convert keyword args to a dictionary of stylesheet parameters.
    XSL stylesheet parameters must be XPath expressions, i.e.:

    * string expressions, like "'5'"
    * simple (number) expressions, like "5"
    * valid XPath expressions, like "/a/b/text()"

    This function converts native Python keyword arguments to stylesheet
    parameters following these rules:
    If an arg is a string wrap it with XSLT.strparam().
    If an arg is an XPath object use its path string.
    If arg is None raise TypeError.
    Else convert arg to string.
    """
    result = {}
    for key, val in kwargs.items():
        if isinstance(val, basestring):
            val = _etree.XSLT.strparam(val)
        elif val is None:
            raise TypeError('None not allowed as a stylesheet parameter')
        elif not isinstance(val, _etree.XPath):
            val = unicode(val)
        result[key] = val
    return result