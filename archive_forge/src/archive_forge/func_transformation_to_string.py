from __future__ import annotations
import re
from fractions import Fraction
def transformation_to_string(matrix, translation_vec=(0, 0, 0), components=('x', 'y', 'z'), c='', delim=','):
    """Convenience method. Given matrix returns string, e.g. x+2y+1/4.

    Args:
        matrix: A 3x3 matrix.
        translation_vec: A 3-element tuple representing the translation vector. Defaults to (0, 0, 0).
        components: A tuple of 3 strings representing the components. Either ('x', 'y', 'z') or ('a', 'b', 'c').
            Defaults to ('x', 'y', 'z').
        c: An optional additional character to print (used for magmoms). Defaults to "".
        delim: A delimiter. Defaults to ",".

    Returns:
        xyz string.
    """
    parts = []
    for idx in range(3):
        string = ''
        m = matrix[idx]
        offset = translation_vec[idx]
        for j, dim in enumerate(components):
            if m[j] != 0:
                f = Fraction(m[j]).limit_denominator()
                if string != '' and f >= 0:
                    string += '+'
                if abs(f.numerator) != 1:
                    string += str(f.numerator)
                elif f < 0:
                    string += '-'
                string += c + dim
                if f.denominator != 1:
                    string += f'/{f.denominator}'
        if offset != 0:
            string += ('+' if offset > 0 and string != '' else '') + str(Fraction(offset).limit_denominator())
        if string == '':
            string += '0'
        parts.append(string)
    return delim.join(parts)