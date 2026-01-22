from __future__ import annotations
from datetime import datetime
from typing import Any
from github.AdvisoryBase import AdvisoryBase
from github.AdvisoryCreditDetailed import AdvisoryCreditDetailed
from github.AdvisoryVulnerability import AdvisoryVulnerability
from github.GithubObject import Attribute, NotSet
@property
def nvd_published_at(self) -> datetime:
    return self._nvd_published_at.value