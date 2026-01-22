from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml, unescapeAll
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def colon_fence_plugin(md: MarkdownIt) -> None:
    """This plugin directly mimics regular fences, but with `:` colons.

    Example::

        :::name
        contained text
        :::

    """
    md.block.ruler.before('fence', 'colon_fence', _rule, {'alt': ['paragraph', 'reference', 'blockquote', 'list', 'footnote_def']})
    md.add_render_rule('colon_fence', _render)