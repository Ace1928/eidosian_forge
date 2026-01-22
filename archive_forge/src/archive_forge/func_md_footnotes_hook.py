import re
from ..core import BlockState
from ..util import unikey
from ..helpers import LINK_LABEL
def md_footnotes_hook(md, result: str, state: BlockState):
    notes = state.env.get('footnotes')
    if not notes:
        return result
    children = [parse_footnote_item(md.block, k, i + 1, state) for i, k in enumerate(notes)]
    state = BlockState()
    state.tokens = [{'type': 'footnotes', 'children': children}]
    output = md.render_state(state)
    return result + output