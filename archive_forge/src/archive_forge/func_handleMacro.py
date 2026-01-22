from pyparsing import (Literal, oneOf, printables, ParserElement, Combine,
def handleMacro(toks):
    macroChar = toks[0][1]
    if macroChar == 'd':
        return CharacterRangeEmitter('0123456789')
    elif macroChar == 'w':
        return CharacterRangeEmitter(srange('[A-Za-z0-9_]'))
    elif macroChar == 's':
        return LiteralEmitter(' ')
    else:
        raise ParseFatalException('', 0, 'unsupported macro character (' + macroChar + ')')