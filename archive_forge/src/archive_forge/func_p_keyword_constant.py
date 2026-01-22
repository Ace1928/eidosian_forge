import types
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
@staticmethod
def p_keyword_constant(p):
    """
        value : KEYWORD_STRING
        """
    p[0] = expressions.KeywordConstant(p[1])