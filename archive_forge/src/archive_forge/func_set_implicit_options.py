from docutils import nodes, utils
from docutils.parsers.rst import directives
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils.code_analyzer import Lexer, LexerError
def set_implicit_options(role_fn):
    """
    Add customization options to role functions, unless explicitly set or
    disabled.
    """
    if not hasattr(role_fn, 'options') or role_fn.options is None:
        role_fn.options = {'class': directives.class_option}
    elif 'class' not in role_fn.options:
        role_fn.options['class'] = directives.class_option