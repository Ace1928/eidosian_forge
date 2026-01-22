import re
import types
from ..util import escape
from ..helpers import PREVENT_BACKSLASH
def parse_ref_abbr(block, m, state):
    ref = state.env.get('ref_abbrs')
    if not ref:
        ref = {}
    key = m.group('abbr_key')
    text = m.group('abbr_text')
    ref[key] = text.strip()
    state.env['ref_abbrs'] = ref
    state.append_token({'type': 'blank_line'})
    return m.end() + 1