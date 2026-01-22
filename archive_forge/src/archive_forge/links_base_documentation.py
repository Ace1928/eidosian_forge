import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet

            Go from a crossing to the mirror, which requires the a rotation of the
            entry points; the direction of rotation depends on the sign of
            the crossing.
            