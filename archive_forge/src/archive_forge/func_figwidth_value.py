import sys
import urllib.request, urllib.parse, urllib.error
from docutils import nodes, utils
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives, states
from docutils.nodes import fully_normalize_name, whitespace_normalize_name
from docutils.parsers.rst.roles import set_classes
def figwidth_value(argument):
    if argument.lower() == 'image':
        return 'image'
    else:
        return directives.length_or_percentage_or_unitless(argument, 'px')