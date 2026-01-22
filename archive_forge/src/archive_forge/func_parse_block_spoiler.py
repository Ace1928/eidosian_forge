import re
def parse_block_spoiler(block, m, state):
    text, end_pos = block.extract_block_quote(m, state)
    if not text.endswith('\n'):
        text += '\n'
    depth = state.depth()
    if not depth and _BLOCK_SPOILER_MATCH.match(text):
        text = _BLOCK_SPOILER_START.sub('', text)
        tok_type = 'block_spoiler'
    else:
        tok_type = 'block_quote'
    child = state.child_state(text)
    if state.depth() >= block.max_nested_level - 1:
        rules = list(block.block_quote_rules)
        rules.remove('block_quote')
    else:
        rules = block.block_quote_rules
    block.parse(child, rules)
    token = {'type': tok_type, 'children': child.tokens}
    if end_pos:
        state.prepend_token(token)
        return end_pos
    state.append_token(token)
    return state.cursor