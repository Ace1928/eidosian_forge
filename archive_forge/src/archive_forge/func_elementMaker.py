from __future__ import print_function
import argparse
import sys
import graphviz
from ._discover import findMachines
def elementMaker(name, *children, **attrs):
    """
    Construct a string from the HTML element description.
    """
    formattedAttrs = ' '.join(('{}={}'.format(key, _gvquote(str(value))) for key, value in sorted(attrs.items())))
    formattedChildren = ''.join(children)
    return u'<{name} {attrs}>{children}</{name}>'.format(name=name, attrs=formattedAttrs, children=formattedChildren)