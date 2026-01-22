from docutils import nodes, utils
from docutils.parsers.rst import directives
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils.code_analyzer import Lexer, LexerError
def code_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    set_classes(options)
    language = options.get('language', '')
    classes = ['code']
    if 'classes' in options:
        classes.extend(options['classes'])
    if language and language not in classes:
        classes.append(language)
    try:
        tokens = Lexer(utils.unescape(text, True), language, inliner.document.settings.syntax_highlight)
    except LexerError as error:
        msg = inliner.reporter.warning(error)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return ([prb], [msg])
    node = nodes.literal(rawtext, '', classes=classes)
    for classes, value in tokens:
        if classes:
            node += nodes.inline(value, value, classes=classes)
        else:
            node += nodes.Text(value, value)
    return ([node], [])