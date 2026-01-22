import types
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
@staticmethod
def p_value_to_dollar(p):
    """
        value : DOLLAR
        """
    p[0] = expressions.GetContextValue(expressions.Constant(p[1]))