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
def columns_url(self) -> str:
    self._completeIfNotSet(self._columns_url)
    return self._columns_url.value