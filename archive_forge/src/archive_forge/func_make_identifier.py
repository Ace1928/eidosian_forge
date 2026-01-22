import operator
def make_identifier(number):
    """
    Encodes a number as an identifier.
    """
    if not isinstance(number, int):
        raise ValueError('You can only make identifiers out of integers (not %r)' % number)
    if number < 0:
        raise ValueError('You cannot make identifiers out of negative numbers: %r' % number)
    result = []
    while number:
        next = number % base
        result.append(good_characters[next])
        number = number // base
    return ''.join(result)