from sympy.assumptions import ask, Q
from sympy.core.basic import Basic
from sympy.core.sympify import _sympify
def make_eval_method(fact):

    def getit(self):
        try:
            pred = getattr(Q, fact)
            ret = ask(pred(self.expr), self.assumptions)
            return ret
        except AttributeError:
            return None
    return getit