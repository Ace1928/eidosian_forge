from __future__ import annotations
import re
from fractions import Fraction
def htmlify(formula: str) -> str:
    """Generates a HTML formatted formula, e.g. Fe2O3 is transformed to
    Fe<sub>2</sub>O</sub>3</sub>.

    Note that Composition now has a to_html_string() method that may
    be used instead.

    Args:
        formula: The string to format.
    """
    return re.sub('([A-Za-z\\(\\)])([\\d\\.]+)', '\\1<sub>\\2</sub>', formula)