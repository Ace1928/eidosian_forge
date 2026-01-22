from __future__ import annotations
from typing import Any
from sympy.core import S, Rational, Float, Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
def indent_code(self, code):
    """Accepts a string of code or a list of code lines"""
    if isinstance(code, str):
        code_lines = self.indent_code(code.splitlines(True))
        return ''.join(code_lines)
    tab = '    '
    inc_token = ('{', '(', '{\n', '(\n')
    dec_token = ('}', ')')
    code = [line.lstrip(' \t') for line in code]
    increase = [int(any(map(line.endswith, inc_token))) for line in code]
    decrease = [int(any(map(line.startswith, dec_token))) for line in code]
    pretty = []
    level = 0
    for n, line in enumerate(code):
        if line in ('', '\n'):
            pretty.append(line)
            continue
        level -= decrease[n]
        pretty.append('%s%s' % (tab * level, line))
        level += increase[n]
    return pretty