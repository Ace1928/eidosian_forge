import types
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
@staticmethod
def p_function(p):
    """
        func : FUNC args ')'
        """
    arg = ()
    if len(p) > 3:
        arg = p[2]
    p[0] = expressions.Function(p[1], *arg)