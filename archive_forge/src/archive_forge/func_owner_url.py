from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
import github.ProjectColumn
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
@property
def owner_url(self) -> str:
    self._completeIfNotSet(self._owner_url)
    return self._owner_url.value