import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def sublink(self, components):
    """
        Returns the sublink consisting of the specified components; see the
        example below for the various accepted forms.

        Warnings: Components in the sublink that are both unknotted
        and unlinked may be silently thrown away.  The order of the
        components in the sublink need not correspond to their order
        in the original link.

        >>> L = Link('L14n64110')
        >>> L
        <Link L14n64110: 5 comp; 14 cross>
        >>> L.sublink([1,2,3,4])
        <Link: 4 comp; 10 cross>
        >>> comps = L.link_components
        >>> L.sublink([comps[0], comps[1]])
        <Link: 2 comp; 2 cross>

        If you just want one component you can do this:

        >>> L11a127 = [(17,9,0,8), (7,12,8,13), (9,17,10,16), (11,3,12,2),
        ... (19,14,20,15), (21,4,18,5), (5,18,6,19), (15,20,16,21), (3,11,4,10),
        ... (1,6,2,7), (13,0,14,1)]
        >>> L = Link(L11a127)
        >>> L.sublink(0)
        <Link: 1 comp; 7 cross>
        >>> L.sublink(L.link_components[1])
        <Link: 0 comp; 0 cross>

        The last answer is empty because the second component is unknotted.
        """
    if components in self.link_components or not is_iterable(components):
        components = [components]
    indices = []
    for c in components:
        if is_iterable(c):
            c = self.link_components.index(c)
        else:
            try:
                self.link_components[c]
            except IndexError:
                raise ValueError('No component of that index')
        indices.append(c)

    def keep(C):
        return all((i in indices for i in C.strand_components))
    L = self.copy()
    final_crossings = []
    for C in L.crossings:
        if keep(C):
            final_crossings.append(C)
        else:
            for j in [0, 1]:
                if C.strand_components[j] in indices:
                    A, a = C.adjacent[j]
                    B, b = C.adjacent[j + 2]
                    A[a] = B[b]
    return type(self)(final_crossings, check_planarity=False)