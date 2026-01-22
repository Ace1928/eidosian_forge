from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Sequence
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def render_myst_target(self: RendererProtocol, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType) -> str:
    label = tokens[idx].content
    class_name = 'myst-target'
    target = f'<a href="#{label}">({label})=</a>'
    return f'<div class="{class_name}">{target}</div>'