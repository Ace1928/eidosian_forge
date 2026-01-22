from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def get_article_template(self, e):
    volume_and_pages = first_of[optional[join[field('volume'), optional['(', field('number'), ')'], ':', pages],], words['pages', pages]]
    template = toplevel[self.format_names('author'), self.format_title(e, 'title'), sentence[tag('em')[field('journal')], optional[volume_and_pages], date], sentence[optional_field('note')], self.format_web_refs(e)]
    return template