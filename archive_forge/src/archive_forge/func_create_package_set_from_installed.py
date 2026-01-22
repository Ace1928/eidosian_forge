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
def create_package_set_from_installed() -> Tuple[PackageSet, bool]:
    """Converts a list of distributions into a PackageSet."""
    package_set = {}
    problems = False
    env = get_default_environment()
    for dist in env.iter_installed_distributions(local_only=False, skip=()):
        name = dist.canonical_name
        try:
            dependencies = list(dist.iter_dependencies())
            package_set[name] = PackageDetails(dist.version, dependencies)
        except (OSError, ValueError) as e:
            logger.warning('Error parsing requirements for %s: %s', name, e)
            problems = True
    return (package_set, problems)