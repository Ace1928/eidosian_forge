from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def describe_tactics():
    """Display a (tabular) description of all available tactics in Z3."""
    if in_html_mode():
        even = True
        print('<table border="1" cellpadding="2" cellspacing="0">')
        for t in tactics():
            if even:
                print('<tr style="background-color:#CFCFCF">')
                even = False
            else:
                print('<tr>')
                even = True
            print('<td>%s</td><td>%s</td></tr>' % (t, insert_line_breaks(tactic_description(t), 40)))
        print('</table>')
    else:
        for t in tactics():
            print('%s : %s' % (t, tactic_description(t)))