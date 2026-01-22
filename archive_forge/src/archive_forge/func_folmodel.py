import inspect
import re
import sys
import textwrap
from pprint import pformat
from nltk.decorators import decorator  # this used in code that is commented out
from nltk.sem.logic import (
def folmodel(quiet=False, trace=None):
    """Example of a first-order model."""
    global val2, v2, dom2, m2, g2
    v2 = [('adam', 'b1'), ('betty', 'g1'), ('fido', 'd1'), ('girl', {'g1', 'g2'}), ('boy', {'b1', 'b2'}), ('dog', {'d1'}), ('love', {('b1', 'g1'), ('b2', 'g2'), ('g1', 'b1'), ('g2', 'b1')})]
    val2 = Valuation(v2)
    dom2 = val2.domain
    m2 = Model(dom2, val2)
    g2 = Assignment(dom2, [('x', 'b1'), ('y', 'g2')])
    if not quiet:
        print()
        print('*' * mult)
        print('Models Demo')
        print('*' * mult)
        print('Model m2:\n', '-' * 14, '\n', m2)
        print('Variable assignment = ', g2)
        exprs = ['adam', 'boy', 'love', 'walks', 'x', 'y', 'z']
        parsed_exprs = [Expression.fromstring(e) for e in exprs]
        print()
        for parsed in parsed_exprs:
            try:
                print("The interpretation of '%s' in m2 is %s" % (parsed, m2.i(parsed, g2)))
            except Undefined:
                print("The interpretation of '%s' in m2 is Undefined" % parsed)
        applications = [('boy', 'adam'), ('walks', ('adam',)), ('love', ('adam', 'y')), ('love', ('y', 'adam'))]
        for fun, args in applications:
            try:
                funval = m2.i(Expression.fromstring(fun), g2)
                argsval = tuple((m2.i(Expression.fromstring(arg), g2) for arg in args))
                print(f'{fun}({args}) evaluates to {argsval in funval}')
            except Undefined:
                print(f'{fun}({args}) evaluates to Undefined')