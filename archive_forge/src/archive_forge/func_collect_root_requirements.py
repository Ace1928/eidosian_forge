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
def collect_root_requirements(self, root_ireqs: List[InstallRequirement]) -> CollectedRootRequirements:
    collected = CollectedRootRequirements([], {}, {})
    for i, ireq in enumerate(root_ireqs):
        if ireq.constraint:
            problem = check_invalid_constraint_type(ireq)
            if problem:
                raise InstallationError(problem)
            if not ireq.match_markers():
                continue
            assert ireq.name, 'Constraint must be named'
            name = canonicalize_name(ireq.name)
            if name in collected.constraints:
                collected.constraints[name] &= ireq
            else:
                collected.constraints[name] = Constraint.from_ireq(ireq)
        else:
            reqs = list(self._make_requirements_from_install_req(ireq, requested_extras=()))
            if not reqs:
                continue
            template = reqs[0]
            if ireq.user_supplied and template.name not in collected.user_requested:
                collected.user_requested[template.name] = i
            collected.requirements.extend(reqs)
    collected.requirements.sort(key=lambda r: r.name != r.project_name)
    return collected