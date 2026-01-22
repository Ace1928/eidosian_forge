from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable
import github.AdvisoryVulnerability
import github.NamedUser
from github.AdvisoryBase import AdvisoryBase
from github.AdvisoryCredit import AdvisoryCredit, Credit
from github.AdvisoryCreditDetailed import AdvisoryCreditDetailed
from github.GithubObject import Attribute, NotSet, Opt
@property
def credits_detailed(self) -> list[AdvisoryCreditDetailed]:
    return self._credits_detailed.value