from ..helpers import iter_sequence
from ..sequence_types import is_sequence_type, match_sequence_type
from ..xpath_tokens import XPathToken, ProxyToken, XPathFunction, XPathMap, XPathArray
from .xpath31_parser import XPath31Parser
@method('map')
def nud_map_sequence_type_or_constructor(self):
    if self.parser.next_token.symbol == '{':
        self.parser.token = XPathMap(self.parser).nud()
        return self.parser.token
    elif self.parser.next_token.symbol != '(':
        return self.as_name()
    self.label = 'kind test'
    self.parser.advance('(')
    if self.parser.next_token.label not in ('kind test', 'sequence type', 'function test'):
        self.parser.expected_next('(name)', ':', '*', message='a QName or a wildcard expected')
    self[:] = (self.parser.expression(45),)
    self[0].parse_occurrence()
    if self[0].symbol != '*':
        self.parser.advance(',')
        if self.parser.next_token.label not in ('kind test', 'sequence type', 'function test'):
            self.parser.expected_next('(name)', ':', '*', message='a QName or a wildcard expected')
        self.append(self.parser.expression(45))
        self[-1].parse_occurrence()
    self.parser.advance(')')
    self.value = None
    return self