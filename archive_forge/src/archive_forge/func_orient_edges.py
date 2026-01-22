from ..sage_helper import _within_sage
from ..graphs import CyclicList, Digraph
from .links import CrossingStrand, Crossing, Strand, Link
from .orthogonal import basic_topological_numbering
from .tangles import join_strands, RationalTangle
def orient_edges(self):
    """
        Orients the edges of the link (that is, its CrossingStrands) with
        respect to the height function.

        sage: L = Link('K3a1')
        sage: D = MorseLinkDiagram(L)
        sage: orients = D.orientations.values()
        sage: len(orients) == 4*len(L.crossings)
        True
        sage: sorted(orients)
        ['down', 'down', 'max', 'max', 'max', 'max', 'min', 'min', 'min', 'min', 'up', 'up']
        sage: list(orients).count('max') == 2*D.morse_number
        True
        """

    def expand_orientation(cs, kind):
        c, i = cs
        kinds = CyclicList(['up', 'down', 'down', 'up'])
        if c.kind in 'horizontal':
            s = 0 if i in [0, 3] else 2
        elif c.kind == 'vertical':
            s = 1 if i in [2, 3] else 3
        if kind in ['down', 'max']:
            s += 2
        return [(CrossingStrand(c, i), kinds[i + s]) for i in range(4)]
    orientations = ImmutableValueDict()
    cs = list(self.bends)[0]
    co = cs.opposite()
    orientations[cs] = 'max'
    orientations[co] = 'max'
    current = [cs, co]
    while len(current):
        new = []
        for cs in current:
            for cn, kind in expand_orientation(cs, orientations[cs]):
                co = cn.opposite()
                if cn in self.bends or co in self.bends:
                    kind = {'up': 'min', 'down': 'max'}[kind]
                if co not in orientations:
                    new.append(co)
                orientations[cn] = kind
                orientations[co] = {'up': 'down', 'down': 'up', 'max': 'max', 'min': 'min'}[kind]
        current = new
    self.orientations = orientations