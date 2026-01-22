import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def withAttribute(*args, **attrDict):
    """Helper to create a validating parse action to be used with start tags created
       with C{L{makeXMLTags}} or C{L{makeHTMLTags}}. Use C{withAttribute} to qualify a starting tag
       with a required attribute value, to avoid false matches on common tags such as
       C{<TD>} or C{<DIV>}.

       Call C{withAttribute} with a series of attribute names and values. Specify the list
       of filter attributes names and values as:
        - keyword arguments, as in C{(align="right")}, or
        - as an explicit dict with C{**} operator, when an attribute name is also a Python
          reserved word, as in C{**{"class":"Customer", "align":"right"}}
        - a list of name-value tuples, as in ( ("ns1:class", "Customer"), ("ns2:align","right") )
       For attribute names with a namespace prefix, you must use the second form.  Attribute
       names are matched insensitive to upper/lower case.

       To verify that the attribute exists, but without specifying a value, pass
       C{withAttribute.ANY_VALUE} as the value.
       """
    if args:
        attrs = args[:]
    else:
        attrs = attrDict.items()
    attrs = [(k, v) for k, v in attrs]

    def pa(s, l, tokens):
        for attrName, attrValue in attrs:
            if attrName not in tokens:
                raise ParseException(s, l, 'no matching attribute ' + attrName)
            if attrValue != withAttribute.ANY_VALUE and tokens[attrName] != attrValue:
                raise ParseException(s, l, "attribute '%s' has value '%s', must be '%s'" % (attrName, tokens[attrName], attrValue))
    return pa