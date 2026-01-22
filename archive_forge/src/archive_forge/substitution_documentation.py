from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from markdown_it.rules_inline import StateInline
from mdit_py_plugins.utils import is_code_block
A plugin to create substitution tokens.

    These, token should be handled by the renderer.

    Example::

        {{ block }}

        a {{ inline }} b

    