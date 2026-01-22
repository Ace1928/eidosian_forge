from ..helpers import iter_sequence
from ..sequence_types import is_sequence_type, match_sequence_type
from ..xpath_tokens import XPathToken, ProxyToken, XPathFunction, XPathMap, XPathArray
from .xpath31_parser import XPath31Parser
@method('=>')
def evaluate_arrow_operator(self, context=None):
    tokens = [self[0]]
    if self[2]:
        tokens.extend(self[2][0].get_argument_tokens())
    func = self[1].get_function(context, arity=len(tokens))
    arguments = [tk.evaluate(context) for tk in tokens]
    return func(*arguments, context=context)