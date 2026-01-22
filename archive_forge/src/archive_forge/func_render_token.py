from ..core import BaseRenderer
from ..util import escape as escape_text, striptags, safe_entity
def render_token(self, token, state):
    func = self._get_method(token['type'])
    attrs = token.get('attrs')
    if 'raw' in token:
        text = token['raw']
    elif 'children' in token:
        text = self.render_tokens(token['children'], state)
    elif attrs:
        return func(**attrs)
    else:
        return func()
    if attrs:
        return func(text, **attrs)
    else:
        return func(text)