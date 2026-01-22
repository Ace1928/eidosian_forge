from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def format_doi(self, e):
    return href[join['https://doi.org/', field('doi', raw=True)], join['doi:', field('doi', raw=True)]]