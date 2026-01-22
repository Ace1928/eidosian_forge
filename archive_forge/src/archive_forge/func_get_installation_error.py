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
def get_installation_error(self, e: 'ResolutionImpossible[Requirement, Candidate]', constraints: Dict[str, Constraint]) -> InstallationError:
    assert e.causes, 'Installation error reported with no cause'
    requires_python_causes = [cause for cause in e.causes if isinstance(cause.requirement, RequiresPythonRequirement) and (not cause.requirement.is_satisfied_by(self._python_candidate))]
    if requires_python_causes:
        return self._report_requires_python_error(cast('Sequence[ConflictCause]', requires_python_causes))
    if len(e.causes) == 1:
        req, parent = e.causes[0]
        if req.name not in constraints:
            return self._report_single_requirement_conflict(req, parent)

    def text_join(parts: List[str]) -> str:
        if len(parts) == 1:
            return parts[0]
        return ', '.join(parts[:-1]) + ' and ' + parts[-1]

    def describe_trigger(parent: Candidate) -> str:
        ireq = parent.get_install_requirement()
        if not ireq or not ireq.comes_from:
            return f'{parent.name}=={parent.version}'
        if isinstance(ireq.comes_from, InstallRequirement):
            return str(ireq.comes_from.name)
        return str(ireq.comes_from)
    triggers = set()
    for req, parent in e.causes:
        if parent is None:
            trigger = req.format_for_error()
        else:
            trigger = describe_trigger(parent)
        triggers.add(trigger)
    if triggers:
        info = text_join(sorted(triggers))
    else:
        info = 'the requested packages'
    msg = f'Cannot install {info} because these package versions have conflicting dependencies.'
    logger.critical(msg)
    msg = '\nThe conflict is caused by:'
    relevant_constraints = set()
    for req, parent in e.causes:
        if req.name in constraints:
            relevant_constraints.add(req.name)
        msg = msg + '\n    '
        if parent:
            msg = msg + f'{parent.name} {parent.version} depends on '
        else:
            msg = msg + 'The user requested '
        msg = msg + req.format_for_error()
    for key in relevant_constraints:
        spec = constraints[key].specifier
        msg += f'\n    The user requested (constraint) {key}{spec}'
    msg = msg + '\n\n' + 'To fix this you could try to:\n' + "1. loosen the range of package versions you've specified\n" + '2. remove package versions to allow pip attempt to solve ' + 'the dependency conflict\n'
    logger.info(msg)
    return DistributionNotFound('ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts')