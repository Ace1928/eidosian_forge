from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
def handleLiteral(toks):
    lit = ''
    for t in toks:
        if t[0] == '\\':
            if t[1] == 't':
                lit += '\t'
            else:
                lit += t[1]
        else:
            lit += t
    return LiteralEmitter(lit)