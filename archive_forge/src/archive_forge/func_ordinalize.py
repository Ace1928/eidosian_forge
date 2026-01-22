import re
import unicodedata
def ordinalize(number):
    """
    Turn a number into an ordinal string used to denote the position in an
    ordered sequence such as 1st, 2nd, 3rd, 4th.

    Examples::

        >>> ordinalize(1)
        '1st'
        >>> ordinalize(2)
        '2nd'
        >>> ordinalize(1002)
        '1002nd'
        >>> ordinalize(1003)
        '1003rd'
        >>> ordinalize(-11)
        '-11th'
        >>> ordinalize(-1021)
        '-1021st'

    """
    return '{}{}'.format(number, ordinal(number))