import io
import math
import os
import typing
import weakref
def integerToRoman(num: int) -> str:
    """Return roman numeral for an integer."""
    roman = ((1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'))

    def roman_num(num):
        for r, ltr in roman:
            x, _ = divmod(num, r)
            yield (ltr * x)
            num -= r * x
            if num <= 0:
                break
    return ''.join([a for a in roman_num(num)])