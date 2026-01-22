import contextlib
import functools
import logging
from typing import (
from pip._vendor.packaging.requirements import InvalidRequirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.resolvelib import ResolutionImpossible
from pip._internal.cache import CacheEntry, WheelCache
from pip._internal.exceptions import (
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution, get_default_environment
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.constructors import (
from pip._internal.req.req_install import (
from pip._internal.resolution.base import InstallRequirementProvider
from pip._internal.utils.compatibility_tags import get_supported
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.packaging import get_requirement
from pip._internal.utils.virtualenv import running_under_virtualenv
from .base import Candidate, CandidateVersion, Constraint, Requirement
from .candidates import (
from .found_candidates import FoundCandidates, IndexCandidateInfo
from .requirements import (
def find_candidates(self, identifier: str, requirements: Mapping[str, Iterable[Requirement]], incompatibilities: Mapping[str, Iterator[Candidate]], constraint: Constraint, prefers_installed: bool) -> Iterable[Candidate]:
    explicit_candidates: Set[Candidate] = set()
    ireqs: List[InstallRequirement] = []
    for req in requirements[identifier]:
        cand, ireq = req.get_candidate_lookup()
        if cand is not None:
            explicit_candidates.add(cand)
        if ireq is not None:
            ireqs.append(ireq)
    with contextlib.suppress(InvalidRequirement):
        parsed_requirement = get_requirement(identifier)
        if parsed_requirement.name != identifier:
            explicit_candidates.update(self._iter_explicit_candidates_from_base(requirements.get(parsed_requirement.name, ()), frozenset(parsed_requirement.extras)))
            for req in requirements.get(parsed_requirement.name, []):
                _, ireq = req.get_candidate_lookup()
                if ireq is not None:
                    ireqs.append(ireq)
    if ireqs:
        try:
            explicit_candidates.update(self._iter_candidates_from_constraints(identifier, constraint, template=ireqs[0]))
        except UnsupportedWheel:
            return ()
    incompat_ids = {id(c) for c in incompatibilities.get(identifier, ())}
    if not explicit_candidates:
        return self._iter_found_candidates(ireqs, constraint.specifier, constraint.hashes, prefers_installed, incompat_ids)
    return (c for c in explicit_candidates if id(c) not in incompat_ids and constraint.is_satisfied_by(c) and all((req.is_satisfied_by(c) for req in requirements[identifier])))