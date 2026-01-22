from __future__ import annotations
from datetime import datetime
from typing import Any
import github.Consts
import github.DeploymentStatus
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
@property
def statuses_url(self) -> str:
    self._completeIfNotSet(self._statuses_url)
    return self._statuses_url.value