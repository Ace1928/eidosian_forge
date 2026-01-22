from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Set
from rdflib.namespace import XSD
def type_promotion(t1: URIRef, t2: Optional[URIRef]) -> URIRef:
    if t2 is None:
        return t1
    t1 = _super_types.get(t1, t1)
    t2 = _super_types.get(t2, t2)
    if t1 == t2:
        return t1
    try:
        if TYPE_CHECKING:
            assert t2 is not None
        return _typePromotionMap[t1][t2]
    except KeyError:
        raise TypeError('Operators cannot combine datatypes %s and %s' % (t1, t2))