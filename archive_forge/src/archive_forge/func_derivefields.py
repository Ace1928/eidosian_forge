from pyparsing import *
def derivefields(t):
    if t.fn_args and t.fn_args[-1] == '...':
        t['varargs'] = True