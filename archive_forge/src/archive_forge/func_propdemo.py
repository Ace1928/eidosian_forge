import inspect
import re
import sys
import textwrap
from pprint import pformat
from nltk.decorators import decorator  # this used in code that is commented out
from nltk.sem.logic import (
def propdemo(trace=None):
    """Example of a propositional model."""
    global val1, dom1, m1, g1
    val1 = Valuation([('P', True), ('Q', True), ('R', False)])
    dom1 = set()
    m1 = Model(dom1, val1)
    g1 = Assignment(dom1)
    print()
    print('*' * mult)
    print('Propositional Formulas Demo')
    print('*' * mult)
    print('(Propositional constants treated as nullary predicates)')
    print()
    print('Model m1:\n', m1)
    print('*' * mult)
    sentences = ['(P & Q)', '(P & R)', '- P', '- R', '- - P', '- (P & R)', '(P | R)', '(R | P)', '(R | R)', '(- P | R)', '(P | - P)', '(P -> Q)', '(P -> R)', '(R -> P)', '(P <-> P)', '(R <-> R)', '(P <-> R)']
    for sent in sentences:
        if trace:
            print()
            m1.evaluate(sent, g1, trace)
        else:
            print(f"The value of '{sent}' is: {m1.evaluate(sent, g1)}")