from __future__ import annotations
from datetime import datetime
from typing import Any
from github.AdvisoryBase import AdvisoryBase
from github.AdvisoryCreditDetailed import AdvisoryCreditDetailed
from github.AdvisoryVulnerability import AdvisoryVulnerability
from github.GithubObject import Attribute, NotSet
@property
def source_code_location(self) -> str:
    return self._source_code_location.value