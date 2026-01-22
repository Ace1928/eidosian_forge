import re
from urllib.parse import quote
from html import _replace_charref
def expand_leading_tab(text: str, width=4):

    def repl(m):
        s = m.group(1)
        return s + ' ' * (width - len(s))
    return _expand_tab_re.sub(repl, text)