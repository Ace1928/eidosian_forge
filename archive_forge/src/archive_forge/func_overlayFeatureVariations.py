from fontTools.misc.dictTools import hashdict
from fontTools.misc.intTools import bit_count
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools.otlLib.builder import buildLookup, buildSingleSubstSubtable
from collections import OrderedDict
from .errors import VarLibError, VarLibValidationError
def overlayFeatureVariations(conditionalSubstitutions):
    """Compute overlaps between all conditional substitutions.

    The `conditionalSubstitutions` argument is a list of (Region, Substitutions)
    tuples.

    A Region is a list of Boxes. A Box is a dict mapping axisTags to
    (minValue, maxValue) tuples. Irrelevant axes may be omitted and they are
    interpretted as extending to end of axis in each direction.  A Box represents
    an orthogonal 'rectangular' subset of an N-dimensional design space.
    A Region represents a more complex subset of an N-dimensional design space,
    ie. the union of all the Boxes in the Region.
    For efficiency, Boxes within a Region should ideally not overlap, but
    functionality is not compromised if they do.

    The minimum and maximum values are expressed in normalized coordinates.

    A Substitution is a dict mapping source glyph names to substitute glyph names.

    Returns data is in similar but different format.  Overlaps of distinct
    substitution Boxes (*not* Regions) are explicitly listed as distinct rules,
    and rules with the same Box merged.  The more specific rules appear earlier
    in the resulting list.  Moreover, instead of just a dictionary of substitutions,
    a list of dictionaries is returned for substitutions corresponding to each
    unique space, with each dictionary being identical to one of the input
    substitution dictionaries.  These dictionaries are not merged to allow data
    sharing when they are converted into font tables.

    Example::

        >>> condSubst = [
        ...     # A list of (Region, Substitution) tuples.
        ...     ([{"wght": (0.5, 1.0)}], {"dollar": "dollar.rvrn"}),
        ...     ([{"wght": (0.5, 1.0)}], {"dollar": "dollar.rvrn"}),
        ...     ([{"wdth": (0.5, 1.0)}], {"cent": "cent.rvrn"}),
        ...     ([{"wght": (0.5, 1.0), "wdth": (-1, 1.0)}], {"dollar": "dollar.rvrn"}),
        ... ]
        >>> from pprint import pprint
        >>> pprint(overlayFeatureVariations(condSubst))
        [({'wdth': (0.5, 1.0), 'wght': (0.5, 1.0)},
          [{'dollar': 'dollar.rvrn'}, {'cent': 'cent.rvrn'}]),
         ({'wdth': (0.5, 1.0)}, [{'cent': 'cent.rvrn'}]),
         ({'wght': (0.5, 1.0)}, [{'dollar': 'dollar.rvrn'}])]

    """
    merged = OrderedDict()
    for value, key in conditionalSubstitutions:
        key = hashdict(key)
        if key in merged:
            merged[key].extend(value)
        else:
            merged[key] = value
    conditionalSubstitutions = [(v, dict(k)) for k, v in merged.items()]
    del merged
    merged = OrderedDict()
    for key, value in reversed(conditionalSubstitutions):
        key = tuple(sorted((hashdict(cleanupBox(k)) for k in key), key=lambda d: tuple(sorted(d.items()))))
        if key in merged:
            merged[key].update(value)
        else:
            merged[key] = dict(value)
    conditionalSubstitutions = list(reversed(merged.items()))
    del merged
    initMapInit = ((hashdict(), 0),)
    boxMap = OrderedDict(initMapInit)
    for i, (currRegion, _) in enumerate(conditionalSubstitutions):
        newMap = OrderedDict(initMapInit)
        currRank = 1 << i
        for box, rank in boxMap.items():
            for currBox in currRegion:
                intersection, remainder = overlayBox(currBox, box)
                if intersection is not None:
                    intersection = hashdict(intersection)
                    newMap[intersection] = newMap.get(intersection, 0) | rank | currRank
                if remainder is not None:
                    remainder = hashdict(remainder)
                    newMap[remainder] = newMap.get(remainder, 0) | rank
        boxMap = newMap
    items = []
    for box, rank in sorted(boxMap.items(), key=lambda BoxAndRank: -bit_count(BoxAndRank[1])):
        if rank == 0:
            continue
        substsList = []
        i = 0
        while rank:
            if rank & 1:
                substsList.append(conditionalSubstitutions[i][1])
            rank >>= 1
            i += 1
        items.append((dict(box), substsList))
    return items