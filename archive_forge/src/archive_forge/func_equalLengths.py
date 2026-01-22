from .pretty_symbology import hobj, vobj, xsym, xobj, pretty_use_unicode, line_width
from sympy.utilities.exceptions import sympy_deprecation_warning
@staticmethod
def equalLengths(lines):
    if not lines:
        return ['']
    width = max((line_width(line) for line in lines))
    return [line.center(width) for line in lines]