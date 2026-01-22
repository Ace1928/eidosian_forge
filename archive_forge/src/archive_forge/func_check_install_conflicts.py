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
def check_install_conflicts(to_install: List[InstallRequirement]) -> ConflictDetails:
    """For checking if the dependency graph would be consistent after     installing given requirements
    """
    package_set, _ = create_package_set_from_installed()
    would_be_installed = _simulate_installation_of(to_install, package_set)
    whitelist = _create_whitelist(would_be_installed, package_set)
    return (package_set, check_package_set(package_set, should_ignore=lambda name: name not in whitelist))