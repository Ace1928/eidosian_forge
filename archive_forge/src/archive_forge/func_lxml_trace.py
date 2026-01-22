import cProfile
from io import BytesIO
from html.parser import HTMLParser
import bs4
from bs4 import BeautifulSoup, __version__
from bs4.builder import builder_registry
import os
import pstats
import random
import tempfile
import time
import traceback
import sys
import cProfile
def lxml_trace(data, html=True, **kwargs):
    """Print out the lxml events that occur during parsing.

    This lets you see how lxml parses a document when no Beautiful
    Soup code is running. You can use this to determine whether
    an lxml-specific problem is in Beautiful Soup's lxml tree builders
    or in lxml itself.

    :param data: Some markup.
    :param html: If True, markup will be parsed with lxml's HTML parser.
       if False, lxml's XML parser will be used.
    """
    from lxml import etree
    recover = kwargs.pop('recover', True)
    if isinstance(data, str):
        data = data.encode('utf8')
    reader = BytesIO(data)
    for event, element in etree.iterparse(reader, html=html, recover=recover, **kwargs):
        print('%s, %4s, %s' % (event, element.tag, element.text))