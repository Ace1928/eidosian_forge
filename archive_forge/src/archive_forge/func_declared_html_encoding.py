from html.entities import codepoint2name
from collections import defaultdict
import codecs
import re
import logging
import string
from html.entities import html5
@property
def declared_html_encoding(self):
    """If the markup is an HTML document, returns the encoding declared _within_
        the document.
        """
    if not self.is_html:
        return None
    return self.detector.declared_encoding