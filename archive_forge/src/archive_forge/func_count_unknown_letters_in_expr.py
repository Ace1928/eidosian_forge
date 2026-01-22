import re
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
from ochat.evaluation.grading import math_normalize
def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace('sqrt', '')
    expr = expr.replace('frac', '')
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)