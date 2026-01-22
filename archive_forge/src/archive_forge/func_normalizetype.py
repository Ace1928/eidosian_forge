from pyparsing import *
def normalizetype(t):
    if isinstance(t, ParseResults):
        return ' '.join(t)