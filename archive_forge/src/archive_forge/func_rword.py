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
def rword(length=5):
    """Generate a random word-like string."""
    s = ''
    for i in range(length):
        if i % 2 == 0:
            t = _consonants
        else:
            t = _vowels
        s += random.choice(t)
    return s