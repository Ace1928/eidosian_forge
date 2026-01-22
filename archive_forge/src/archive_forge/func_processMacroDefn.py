from pyparsing import *
def processMacroDefn(s, l, t):
    macroVal = macroExpander.transformString(t.value)
    macros[t.macro] = macroVal
    macroExpr << MatchFirst(map(Keyword, macros.keys()))
    return '#def ' + t.macro + ' ' + macroVal