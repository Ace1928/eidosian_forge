from __future__ import annotations
from datetime import datetime
from typing import Any
from github.AdvisoryBase import AdvisoryBase
from github.AdvisoryCreditDetailed import AdvisoryCreditDetailed
from github.AdvisoryVulnerability import AdvisoryVulnerability
from github.GithubObject import Attribute, NotSet
@property
def github_reviewed_at(self) -> datetime:
    return self._github_reviewed_at.value