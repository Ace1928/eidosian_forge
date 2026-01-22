import difflib
import re
from collections import defaultdict
from html.parser import HTMLParser
import bokeh.core.properties as bp
from bokeh.events import ModelEvent
from bokeh.model import DataModel
from bokeh.models import LayoutDOM
from .layout import HTMLBox
def find_attrs(html):
    p = ReactiveHTMLParser()
    p.feed(html)
    return p.attrs