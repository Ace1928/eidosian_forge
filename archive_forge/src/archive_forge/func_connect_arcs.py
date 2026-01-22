from ..snap.t3mlite.simplex import *
from ..snap.t3mlite.edge import Edge
from ..snap.t3mlite.arrow import Arrow
from ..snap.t3mlite.tetrahedron import Tetrahedron
from ..snap.t3mlite.mcomplex import VERBOSE
from .exceptions import GeneralPositionError
from .rational_linear_algebra import Vector3, QQ
from . import pl_utils
from . import stored_moves
from .mcomplex_with_expansion import McomplexWithExpansion
from .mcomplex_with_memory import McomplexWithMemory
from .barycentric_geometry import (BarycentricPoint, BarycentricArc,
import random
import collections
import time
def connect_arcs(self, tetrahedra=None):
    if tetrahedra is None:
        tetrahedra = self.Tetrahedra
    for tet in tetrahedra:
        on_faces = collections.Counter()
        for arc in tet.arcs:
            arc.tet = tet
            if not isinstance(arc, list):
                if arc.past is None or arc.next is None:
                    for other_arc in tet.arcs:
                        if arc.end == other_arc.start:
                            arc.glue_to(other_arc)
                        elif other_arc.end == arc.start:
                            other_arc.glue_to(arc)
                        elif arc.past is not None and arc.next is not None:
                            break
            on_faces.update([pt for pt in [arc.start, arc.end] if pt.on_boundary()])
        if max(on_faces.values(), default=0) > 1:
            raise ValueError('Houston, we have a bounce')
    if tetrahedra == self.Tetrahedra:
        faces = self.Faces
    else:
        faces = {tet.Class[F] for tet in tetrahedra for F in TwoSubsimplices}
    for face in faces:
        for x, y in pair_arcs_across_face(face):
            between = InfinitesimalArc(x.end, y.start, x.tet, y.tet, past=x, next=y)
            x.next = between
            y.past = between