import collections
import math
from typing import (
from pip._vendor.resolvelib.providers import AbstractProvider
from .base import Candidate, Constraint, Requirement
from .candidates import REQUIRES_PYTHON_IDENTIFIER
from .factory import Factory
def get_preference(self, identifier: str, resolutions: Mapping[str, Candidate], candidates: Mapping[str, Iterator[Candidate]], information: Mapping[str, Iterable['PreferenceInformation']], backtrack_causes: Sequence['PreferenceInformation']) -> 'Preference':
    """Produce a sort key for given requirement based on preference.

        The lower the return value is, the more preferred this group of
        arguments is.

        Currently pip considers the following in order:

        * Prefer if any of the known requirements is "direct", e.g. points to an
          explicit URL.
        * If equal, prefer if any requirement is "pinned", i.e. contains
          operator ``===`` or ``==``.
        * If equal, calculate an approximate "depth" and resolve requirements
          closer to the user-specified requirements first. If the depth cannot
          by determined (eg: due to no matching parents), it is considered
          infinite.
        * Order user-specified requirements by the order they are specified.
        * If equal, prefers "non-free" requirements, i.e. contains at least one
          operator, such as ``>=`` or ``<``.
        * If equal, order alphabetically for consistency (helps debuggability).
        """
    try:
        next(iter(information[identifier]))
    except StopIteration:
        has_information = False
    else:
        has_information = True
    if has_information:
        lookups = (r.get_candidate_lookup() for r, _ in information[identifier])
        candidate, ireqs = zip(*lookups)
    else:
        candidate, ireqs = (None, ())
    operators = [specifier.operator for specifier_set in (ireq.specifier for ireq in ireqs if ireq) for specifier in specifier_set]
    direct = candidate is not None
    pinned = any((op[:2] == '==' for op in operators))
    unfree = bool(operators)
    try:
        requested_order: Union[int, float] = self._user_requested[identifier]
    except KeyError:
        requested_order = math.inf
        if has_information:
            parent_depths = (self._known_depths[parent.name] if parent is not None else 0.0 for _, parent in information[identifier])
            inferred_depth = min((d for d in parent_depths)) + 1.0
        else:
            inferred_depth = math.inf
    else:
        inferred_depth = 1.0
    self._known_depths[identifier] = inferred_depth
    requested_order = self._user_requested.get(identifier, math.inf)
    requires_python = identifier == REQUIRES_PYTHON_IDENTIFIER
    backtrack_cause = self.is_backtrack_cause(identifier, backtrack_causes)
    return (not requires_python, not direct, not pinned, not backtrack_cause, inferred_depth, requested_order, not unfree, identifier)