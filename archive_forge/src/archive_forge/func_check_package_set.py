import logging
from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import LegacySpecifier
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.packaging.version import LegacyVersion
from pip._internal.distributions import make_distribution_for_install_requirement
from pip._internal.metadata import get_default_environment
from pip._internal.metadata.base import DistributionVersion
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.deprecation import deprecated
def check_package_set(package_set: PackageSet, should_ignore: Optional[Callable[[str], bool]]=None) -> CheckResult:
    """Check if a package set is consistent

    If should_ignore is passed, it should be a callable that takes a
    package name and returns a boolean.
    """
    warn_legacy_versions_and_specifiers(package_set)
    missing = {}
    conflicting = {}
    for package_name, package_detail in package_set.items():
        missing_deps: Set[Missing] = set()
        conflicting_deps: Set[Conflicting] = set()
        if should_ignore and should_ignore(package_name):
            continue
        for req in package_detail.dependencies:
            name = canonicalize_name(req.name)
            if name not in package_set:
                missed = True
                if req.marker is not None:
                    missed = req.marker.evaluate({'extra': ''})
                if missed:
                    missing_deps.add((name, req))
                continue
            version = package_set[name].version
            if not req.specifier.contains(version, prereleases=True):
                conflicting_deps.add((name, version, req))
        if missing_deps:
            missing[package_name] = sorted(missing_deps, key=str)
        if conflicting_deps:
            conflicting[package_name] = sorted(conflicting_deps, key=str)
    return (missing, conflicting)