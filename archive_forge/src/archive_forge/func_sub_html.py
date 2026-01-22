import re
import sys
from html import escape
from urllib.parse import quote
from paste.util.looper import looper
def sub_html(content, **kw):
    name = kw.get('__name')
    tmpl = HTMLTemplate(content, name=name)
    return tmpl.substitute(kw)