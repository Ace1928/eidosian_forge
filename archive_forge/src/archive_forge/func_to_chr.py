from pyparsing import Literal, CaselessLiteral, Word, delimitedList \
def to_chr(x):
    """chr(x) if 0 < x < 128 ; unicode(x) if x > 127."""
    return 0 < x < 128 and chr(x) or eval("u'\\u%d'" % x)