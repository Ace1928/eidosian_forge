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
def rdoc(num_elements=1000):
    """Randomly generate an invalid HTML document."""
    tag_names = ['p', 'div', 'span', 'i', 'b', 'script', 'table']
    elements = []
    for i in range(num_elements):
        choice = random.randint(0, 3)
        if choice == 0:
            tag_name = random.choice(tag_names)
            elements.append('<%s>' % tag_name)
        elif choice == 1:
            elements.append(rsentence(random.randint(1, 4)))
        elif choice == 2:
            tag_name = random.choice(tag_names)
            elements.append('</%s>' % tag_name)
    return '<html>' + '\n'.join(elements) + '</html>'