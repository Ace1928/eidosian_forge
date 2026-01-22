from .pretty_symbology import hobj, vobj, xsym, xobj, pretty_use_unicode, line_width
from sympy.utilities.exceptions import sympy_deprecation_warning
def parens(self, left='(', right=')', ifascii_nougly=False):
    """Put parentheses around self.
        Returns string, baseline arguments for stringPict.

        left or right can be None or empty string which means 'no paren from
        that side'
        """
    h = self.height()
    b = self.baseline
    if ifascii_nougly and (not pretty_use_unicode()):
        h = 1
        b = 0
    res = self
    if left:
        lparen = stringPict(vobj(left, h), baseline=b)
        res = stringPict(*lparen.right(self))
    if right:
        rparen = stringPict(vobj(right, h), baseline=b)
        res = stringPict(*res.right(rparen))
    return ('\n'.join(res.picture), res.baseline)