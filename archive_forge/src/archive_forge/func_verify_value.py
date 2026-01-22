from pyparsing import *
def verify_value(s, tokens):
    expected = roman_int_map[s]
    if tokens[0] != expected:
        raise Exception('incorrect value for {0} ({1}), expected {2}'.format(s, tokens[0], expected))