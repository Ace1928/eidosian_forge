from .arrow import eArrow
from .simplex import *
from .tetrahedron import Tetrahedron
import re
def read_edge(edge):
    m = re.match('([0-9]+)([uvwx])([uvwx])', edge)
    return (int(m.group(1)) - 1, conv[m.group(2)], conv[m.group(3)])