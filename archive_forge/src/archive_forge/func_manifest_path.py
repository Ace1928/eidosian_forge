from __future__ import annotations
from typing import Any
from github.AdvisoryVulnerabilityPackage import AdvisoryVulnerabilityPackage
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def manifest_path(self) -> str:
    return self._manifest_path.value