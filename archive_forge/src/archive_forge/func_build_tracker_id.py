import logging
from typing import Iterable, Optional, Set, Tuple
from pip._internal.build_env import BuildEnvironment
from pip._internal.distributions.base import AbstractDistribution
from pip._internal.exceptions import InstallationError
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution
from pip._internal.utils.subprocess import runner_with_spinner_message
@property
def build_tracker_id(self) -> Optional[str]:
    """Identify this requirement uniquely by its link."""
    assert self.req.link
    return self.req.link.url_without_fragment