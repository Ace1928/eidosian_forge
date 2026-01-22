import types
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
@staticmethod
def p_named_arg_definition(p):
    """
        named_arg : value MAPPING value
        """
    p[0] = expressions.MappingRuleExpression(p[1], p[3])