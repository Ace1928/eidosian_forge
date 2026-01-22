from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.CheckRunAnnotation
import github.CheckRunOutput
import github.GithubApp
import github.GithubObject
import github.PullRequest
from github.GithubObject import (
from github.PaginatedList import PaginatedList
@property
def details_url(self) -> str:
    self._completeIfNotSet(self._details_url)
    return self._details_url.value