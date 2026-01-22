from __future__ import annotations
import re
from typing import TYPE_CHECKING, Callable, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def render_amsmath_block(self: RendererProtocol, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType) -> str:
    content = _renderer(str(tokens[idx].content))
    return f'<div class="math amsmath">\n{content}\n</div>\n'