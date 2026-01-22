from pyparsing import *
def do_meta_identifier(str, loc, toks):
    if toks[0] in symbol_table:
        return symbol_table[toks[0]]
    else:
        forward_count.value += 1
        symbol_table[toks[0]] = Forward()
        return symbol_table[toks[0]]