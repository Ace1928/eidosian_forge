from .arrow import eArrow
from .simplex import *
from .tetrahedron import Tetrahedron
import re
def write_geo_file(mcomplex, fileobject):
    out = fileobject.write
    out('k\n')
    i = 1
    for edge in mcomplex.Edges:
        tet = edge.Corners[0].Tetrahedron
        edge_name = edge.Corners[0].Subsimplex
        init = Head[edge_name]
        fin = Tail[edge_name]
        a = eArrow(tet, init, fin).opposite()
        b = a.copy()
        out('%d\t%d%s%s ' % (i, mcomplex.Tetrahedra.index(b.Tetrahedron) + 1, conv_back[b.tail()], conv_back[b.head()]))
        b.next()
        while b != a:
            out('%d%s%s ' % (mcomplex.Tetrahedra.index(b.Tetrahedron) + 1, conv_back[b.tail()], conv_back[b.head()]))
            b.next()
        i = i + 1
        out('\n')