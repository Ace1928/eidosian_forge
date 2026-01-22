import re
from ._base import DirectivePlugin
from ..util import escape as escape_text, escape_url
def parse_directive_content(self, block, m, state):
    content = self.parse_content(m)
    if not content:
        return
    tokens = self.parse_tokens(block, content, state)
    caption = tokens[0]
    if caption['type'] == 'paragraph':
        caption['type'] = 'figcaption'
        children = [caption]
        if len(tokens) > 1:
            children.append({'type': 'legend', 'children': tokens[1:]})
        return children