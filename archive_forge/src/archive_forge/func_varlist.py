import os
import textwrap
from passlib.utils.compat import irange
def varlist(name, count):
    return ', '.join((name + str(x) for x in irange(count)))