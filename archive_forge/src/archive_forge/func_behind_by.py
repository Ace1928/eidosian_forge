from __future__ import annotations
from typing import Any
import github.Commit
import github.File
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
from github.PaginatedList import PaginatedList
@property
def behind_by(self) -> int:
    self._completeIfNotSet(self._behind_by)
    return self._behind_by.value