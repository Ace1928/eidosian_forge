from __future__ import annotations
from datetime import datetime
from typing import Any
import github.CodeScanAlertInstance
import github.CodeScanRule
import github.CodeScanTool
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
from github.PaginatedList import PaginatedList
@property
def dismissed_at(self) -> datetime | None:
    return self._dismissed_at.value