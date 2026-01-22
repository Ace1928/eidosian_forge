from docutils import nodes, utils
from docutils.parsers.rst import directives
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils.code_analyzer import Lexer, LexerError
def register_generic_role(canonical_name, node_class):
    """For roles which simply wrap a given `node_class` around the text."""
    role = GenericRole(canonical_name, node_class)
    register_canonical_role(canonical_name, role)