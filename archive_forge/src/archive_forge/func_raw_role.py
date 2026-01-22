from docutils import nodes, utils
from docutils.parsers.rst import directives
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils.code_analyzer import Lexer, LexerError
def raw_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    if not inliner.document.settings.raw_enabled:
        msg = inliner.reporter.warning('raw (and derived) roles disabled')
        prb = inliner.problematic(rawtext, rawtext, msg)
        return ([prb], [msg])
    if 'format' not in options:
        msg = inliner.reporter.error('No format (Writer name) is associated with this role: "%s".\nThe "raw" role cannot be used directly.\nInstead, use the "role" directive to create a new role with an associated format.' % role, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return ([prb], [msg])
    set_classes(options)
    node = nodes.raw(rawtext, utils.unescape(text, True), **options)
    node.source, node.line = inliner.reporter.get_source_and_line(lineno)
    return ([node], [])