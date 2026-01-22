import re
def parse_method(self, block, m, state):
    _type = self.parser.parse_type(m)
    method = self._methods.get(_type)
    if method:
        try:
            token = method(block, m, state)
        except ValueError as e:
            token = {'type': 'block_error', 'raw': str(e)}
    else:
        text = m.group(0)
        token = {'type': 'block_error', 'raw': text}
    if isinstance(token, list):
        for tok in token:
            state.append_token(tok)
    else:
        state.append_token(token)
    return token