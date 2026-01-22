import re
from urllib.parse import quote
from html import _replace_charref
def expand_tab(text: str, space: str='    '):
    repl = '\\1' + space
    return _expand_tab_re.sub(repl, text)