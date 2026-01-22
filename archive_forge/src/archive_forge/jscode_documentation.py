from __future__ import annotations
from typing import Any
from sympy.core import S
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
Accepts a string of code or a list of code lines