from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
def optionalGen():
    yield ''
    for s in self.expr.makeGenerator()():
        yield s