from docstring_parser.common import DocstringReturns
from docstring_parser.util import combine_docstrings
def fun1(arg_a, arg_b, arg_c, arg_d):
    """short_description: fun1

        :param arg_a: fun1
        :param arg_b: fun1
        :return: fun1
        """
    assert arg_a and arg_b and arg_c and arg_d