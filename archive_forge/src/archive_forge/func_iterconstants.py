from __future__ import division, absolute_import
from functools import partial
from itertools import count
from operator import and_, or_, xor
@classmethod
def iterconstants(cls):
    """
        Iteration over a L{Names} subclass results in all of the constants it
        contains.

        @return: an iterator the elements of which are the L{NamedConstant}
            instances defined in the body of this L{Names} subclass.
        """
    constants = cls._enumerants.values()
    return iter(sorted(constants, key=lambda descriptor: descriptor._index))