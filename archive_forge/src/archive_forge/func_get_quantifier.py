import re
from django.utils.functional import SimpleLazyObject
def get_quantifier(ch, input_iter):
    """
    Parse a quantifier from the input, where "ch" is the first character in the
    quantifier.

    Return the minimum number of occurrences permitted by the quantifier and
    either None or the next character from the input_iter if the next character
    is not part of the quantifier.
    """
    if ch in '*?+':
        try:
            ch2, escaped = next(input_iter)
        except StopIteration:
            ch2 = None
        if ch2 == '?':
            ch2 = None
        if ch == '+':
            return (1, ch2)
        return (0, ch2)
    quant = []
    while ch != '}':
        ch, escaped = next(input_iter)
        quant.append(ch)
    quant = quant[:-1]
    values = ''.join(quant).split(',')
    try:
        ch, escaped = next(input_iter)
    except StopIteration:
        ch = None
    if ch == '?':
        ch = None
    return (int(values[0]), ch)