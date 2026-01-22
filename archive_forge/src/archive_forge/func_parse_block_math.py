def parse_block_math(block, m, state):
    text = m.group('math_text')
    state.append_token({'type': 'block_math', 'raw': text})
    return m.end() + 1