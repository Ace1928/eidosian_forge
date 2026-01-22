from ..helpers import iter_sequence
from ..sequence_types import is_sequence_type, match_sequence_type
from ..xpath_tokens import XPathToken, ProxyToken, XPathFunction, XPathMap, XPathArray
from .xpath31_parser import XPath31Parser
@method('=>', bp=67)
def led_arrow_operator(self, left):
    next_token = self.parser.next_token
    if next_token.symbol == '$':
        self[:] = (left, self.parser.expression(80))
    elif isinstance(next_token, ProxyToken):
        self.parser.parse_arguments = False
        self[:] = (left, next_token.nud())
        self.parser.parse_arguments = True
        self.parser.advance()
    elif isinstance(next_token, XPathFunction):
        self[:] = (left, next_token)
        self.parser.advance()
    else:
        next_token.expected('(name)', ':', 'Q{', '(')
        self.parser.parse_arguments = False
        self[:] = (left, self.parser.expression(80))
        self.parser.parse_arguments = True
    right = self.parser.expression(67)
    right.expected('(')
    self.append(right)
    return self