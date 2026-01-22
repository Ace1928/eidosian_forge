from pyparsing import \
import pprint
def inifile_BNF():
    global inibnf
    if not inibnf:
        lbrack = Literal('[').suppress()
        rbrack = Literal(']').suppress()
        equals = Literal('=').suppress()
        semi = Literal(';')
        comment = semi + Optional(restOfLine)
        nonrbrack = ''.join([c for c in printables if c != ']']) + ' \t'
        nonequals = ''.join([c for c in printables if c != '=']) + ' \t'
        sectionDef = lbrack + Word(nonrbrack) + rbrack
        keyDef = ~lbrack + Word(nonequals) + equals + empty + restOfLine

        def stripKey(tokens):
            tokens[0] = tokens[0].strip()
        keyDef.setParseAction(stripKey)
        inibnf = Dict(ZeroOrMore(Group(sectionDef + Dict(ZeroOrMore(Group(keyDef))))))
        inibnf.ignore(comment)
    return inibnf