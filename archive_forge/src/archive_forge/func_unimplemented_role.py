from docutils import nodes, utils
from docutils.parsers.rst import directives
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils.code_analyzer import Lexer, LexerError
def unimplemented_role(role, rawtext, text, lineno, inliner, attributes={}):
    msg = inliner.reporter.error('Interpreted text role "%s" not implemented.' % role, line=lineno)
    prb = inliner.problematic(rawtext, rawtext, msg)
    return ([prb], [msg])