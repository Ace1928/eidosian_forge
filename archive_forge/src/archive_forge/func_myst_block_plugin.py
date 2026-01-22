from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Sequence
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def myst_block_plugin(md: MarkdownIt) -> None:
    """Parse MyST targets (``(name)=``), blockquotes (``% comment``) and block breaks (``+++``)."""
    md.block.ruler.before('blockquote', 'myst_line_comment', line_comment, {'alt': ['paragraph', 'reference', 'blockquote', 'list', 'footnote_def']})
    md.block.ruler.before('hr', 'myst_block_break', block_break, {'alt': ['paragraph', 'reference', 'blockquote', 'list', 'footnote_def']})
    md.block.ruler.before('hr', 'myst_target', target, {'alt': ['paragraph', 'reference', 'blockquote', 'list', 'footnote_def']})
    md.add_render_rule('myst_target', render_myst_target)
    md.add_render_rule('myst_line_comment', render_myst_line_comment)