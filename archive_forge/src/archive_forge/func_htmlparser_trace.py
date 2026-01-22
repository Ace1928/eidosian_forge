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
def htmlparser_trace(data):
    """Print out the HTMLParser events that occur during parsing.

    This lets you see how HTMLParser parses a document when no
    Beautiful Soup code is running.

    :param data: Some markup.
    """
    parser = AnnouncingParser()
    parser.feed(data)