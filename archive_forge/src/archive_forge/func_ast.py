import collections
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
def ast(self):
    """AST representation."""
    if self.has_subscript():
        return gast.Subscript(value=self.parent.ast(), slice=self.qn[-1].ast(), ctx=CallerMustSetThis)
    if self.has_attr():
        return gast.Attribute(value=self.parent.ast(), attr=self.qn[-1], ctx=CallerMustSetThis)
    base = self.qn[0]
    if isinstance(base, str):
        return gast.Name(base, ctx=CallerMustSetThis, annotation=None, type_comment=None)
    elif isinstance(base, Literal):
        return gast.Constant(base.value, kind=None)
    else:
        assert False, 'the constructor should prevent types other than str and Literal'