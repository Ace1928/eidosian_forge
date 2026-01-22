from __future__ import annotations
import warnings
from functools import total_ordering
from typing import (
from rdflib.term import Node, URIRef
def evalPath(graph: Graph, t: Tuple[Optional['_SubjectType'], Union[None, Path, _PredicateType], Optional['_ObjectType']]) -> Iterator[Tuple[_SubjectType, _ObjectType]]:
    warnings.warn(DeprecationWarning('rdflib.path.evalPath() is deprecated, use the (snake-cased) eval_path(). The mixed-case evalPath() function name is incompatible with PEP8 recommendations and will be replaced by eval_path() in rdflib 7.0.0.'))
    return eval_path(graph, t)