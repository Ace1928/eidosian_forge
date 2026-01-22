import types
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
@staticmethod
def p_named_arg_list(p):
    """
        named_arglist : named_arg
                      | named_arglist ',' named_arg
        """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]