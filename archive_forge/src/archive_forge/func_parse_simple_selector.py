import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
def parse_simple_selector(stream: 'TokenStream', inside_negation: bool=False) -> Tuple[Tree, Optional[PseudoElement]]:
    stream.skip_whitespace()
    selector_start = len(stream.used)
    peek = stream.peek()
    if peek.type == 'IDENT' or peek == ('DELIM', '*'):
        if peek.type == 'IDENT':
            namespace = stream.next().value
        else:
            stream.next()
            namespace = None
        if stream.peek() == ('DELIM', '|'):
            stream.next()
            element = stream.next_ident_or_star()
        else:
            element = namespace
            namespace = None
    else:
        element = namespace = None
    result: Tree = Element(namespace, element)
    pseudo_element: Optional[PseudoElement] = None
    while 1:
        peek = stream.peek()
        if peek.type in ('S', 'EOF') or peek.is_delim(',', '+', '>', '~') or (inside_negation and peek == ('DELIM', ')')):
            break
        if pseudo_element:
            raise SelectorSyntaxError('Got pseudo-element ::%s not at the end of a selector' % pseudo_element)
        if peek.type == 'HASH':
            result = Hash(result, typing.cast(str, stream.next().value))
        elif peek == ('DELIM', '.'):
            stream.next()
            result = Class(result, stream.next_ident())
        elif peek == ('DELIM', '|'):
            stream.next()
            result = Element(None, stream.next_ident())
        elif peek == ('DELIM', '['):
            stream.next()
            result = parse_attrib(result, stream)
        elif peek == ('DELIM', ':'):
            stream.next()
            if stream.peek() == ('DELIM', ':'):
                stream.next()
                pseudo_element = stream.next_ident()
                if stream.peek() == ('DELIM', '('):
                    stream.next()
                    pseudo_element = FunctionalPseudoElement(pseudo_element, parse_arguments(stream))
                continue
            ident = stream.next_ident()
            if ident.lower() in ('first-line', 'first-letter', 'before', 'after'):
                pseudo_element = str(ident)
                continue
            if stream.peek() != ('DELIM', '('):
                result = Pseudo(result, ident)
                if repr(result) == 'Pseudo[Element[*]:scope]':
                    if not (len(stream.used) == 2 or (len(stream.used) == 3 and stream.used[0].type == 'S') or (len(stream.used) >= 3 and stream.used[-3].is_delim(',')) or (len(stream.used) >= 4 and stream.used[-3].type == 'S' and stream.used[-4].is_delim(','))):
                        raise SelectorSyntaxError('Got immediate child pseudo-element ":scope" not at the start of a selector')
                continue
            stream.next()
            stream.skip_whitespace()
            if ident.lower() == 'not':
                if inside_negation:
                    raise SelectorSyntaxError('Got nested :not()')
                argument, argument_pseudo_element = parse_simple_selector(stream, inside_negation=True)
                next = stream.next()
                if argument_pseudo_element:
                    raise SelectorSyntaxError('Got pseudo-element ::%s inside :not() at %s' % (argument_pseudo_element, next.pos))
                if next != ('DELIM', ')'):
                    raise SelectorSyntaxError("Expected ')', got %s" % (next,))
                result = Negation(result, argument)
            elif ident.lower() == 'has':
                combinator, arguments = parse_relative_selector(stream)
                result = Relation(result, combinator, arguments)
            elif ident.lower() in ('matches', 'is'):
                selectors = parse_simple_selector_arguments(stream)
                result = Matching(result, selectors)
            elif ident.lower() == 'where':
                selectors = parse_simple_selector_arguments(stream)
                result = SpecificityAdjustment(result, selectors)
            else:
                result = Function(result, ident, parse_arguments(stream))
        else:
            raise SelectorSyntaxError('Expected selector, got %s' % (peek,))
    if len(stream.used) == selector_start:
        raise SelectorSyntaxError('Expected selector, got %s' % (stream.peek(),))
    return (result, pseudo_element)