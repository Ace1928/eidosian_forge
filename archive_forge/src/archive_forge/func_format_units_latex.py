import copy
import operator
import re
import threading
def format_units_latex(udict, font='mathrm', mult='\\\\cdot', paren=False):
    """
    Replace the units string provided with an equivalent latex string.

    Division (a/b) will be replaced by \x0crac{a}{b}.

    Exponentiation (m**2) will be replaced with superscripts (m^{2})

    The latex is set  with the font argument, and the default is the normal,
    non-italicized font mathrm.  Other useful options include 'mathnormal',
    'mathit', 'mathsf', and 'mathtt'.

    Multiplication (*) are replaced with the symbol specified by the mult argument.
    By default this is the latex \\cdot symbol.  Other useful
    options may be '' or '*'.

    If paren=True, encapsulate the string in '\\left(' and '\\right)'

    The result of format_units_latex is encapsulated in $.  This allows the result
    to be used directly in Latex in normal text mode, or in Matplotlib text via the
    MathText feature.

    Restrictions:
    This routine will not put CompoundUnits into a fractional form.
    """
    res = format_units(udict)
    if res.startswith('(') and res.endswith(')'):
        compound = True
    else:
        compound = False
        res = re.sub('(?P<num>.+)/(?P<den>.+)', '\\\\frac{\\g<num>}{\\g<den>}', res)
    res = re.sub('\\*{2,2}(?P<exp>\\d+)', '^{\\g<exp>}', res)
    res = re.sub('\\*', '{' + mult + '}', res)
    if paren and (not compound):
        res = '\\left(%s\\right)' % res
    res = f'$\\{font}{{{res}}}$'
    return res