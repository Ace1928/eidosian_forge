from __future__ import annotations
from datetime import datetime
from typing import Any
from github.AdvisoryBase import AdvisoryBase
from github.AdvisoryCreditDetailed import AdvisoryCreditDetailed
from github.AdvisoryVulnerability import AdvisoryVulnerability
from github.GithubObject import Attribute, NotSet
@property
def repository_advisory_url(self) -> str:
    return self._repository_advisory_url.value