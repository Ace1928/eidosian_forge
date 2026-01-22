from ..helpers import iter_sequence
from ..sequence_types import is_sequence_type, match_sequence_type
from ..xpath_tokens import XPathToken, ProxyToken, XPathFunction, XPathMap, XPathArray
from .xpath31_parser import XPath31Parser
@method('[')
def nud_square_array_constructor(self):
    if self.parser.version < '3.1':
        raise self.wrong_syntax()
    token = XPathArray(self.parser)
    token.symbol = '['
    if token.parser.next_token.symbol not in (']', '(end)'):
        while True:
            token.append(self.parser.expression(5))
            if token.parser.next_token.symbol != ',':
                break
            token.parser.advance()
    token.parser.advance(']')
    return token