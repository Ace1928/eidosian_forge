from __future__ import annotations
import re
from fractions import Fraction
def unicodeify_species(specie_string):
    """Generates a unicode formatted species string, with appropriate
    superscripts for oxidation states.

    Note that Species now has a to_unicode_string() method that
    may be used instead.

    Args:
        specie_string (str): Species string, e.g. O2-

    Returns:
        Species string, e.g. O²⁻
    """
    if not specie_string:
        return ''
    for char, unicode_char in SUPERSCRIPT_UNICODE.items():
        specie_string = specie_string.replace(char, unicode_char)
    return specie_string