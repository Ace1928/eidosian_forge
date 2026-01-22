from typing import Dict, Any
from textwrap import indent
from ._list import render_list
from ..core import BaseRenderer, BlockState
from ..util import strip_end
def block_text(self, token: Dict[str, Any], state: BlockState) -> str:
    return self.render_children(token, state) + '\n'