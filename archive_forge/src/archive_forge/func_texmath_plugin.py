from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any, Callable, Match, Sequence, TypedDict
from markdown_it import MarkdownIt
from markdown_it.common.utils import charCodeAt
def texmath_plugin(md: MarkdownIt, delimiters: str='dollars', macros: Any=None) -> None:
    """Plugin ported from
    `markdown-it-texmath <https://github.com/goessner/markdown-it-texmath>`__.

    It parses TeX math equations set inside opening and closing delimiters:

    .. code-block:: md

        $\\alpha = \\frac{1}{2}$

    :param delimiters: one of: brackets, dollars, gitlab, julia, kramdown

    """
    macros = macros or {}
    if delimiters in rules:
        for rule_inline in rules[delimiters]['inline']:
            md.inline.ruler.before('escape', rule_inline['name'], make_inline_func(rule_inline))

            def render_math_inline(self: RendererProtocol, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType) -> str:
                return rule_inline['tmpl'].format(render(tokens[idx].content, False, macros))
            md.add_render_rule(rule_inline['name'], render_math_inline)
        for rule_block in rules[delimiters]['block']:
            md.block.ruler.before('fence', rule_block['name'], make_block_func(rule_block))

            def render_math_block(self: RendererProtocol, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType) -> str:
                return rule_block['tmpl'].format(render(tokens[idx].content, True, macros), tokens[idx].info)
            md.add_render_rule(rule_block['name'], render_math_block)