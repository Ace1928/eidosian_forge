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
def external_id(self) -> str:
    self._completeIfNotSet(self._external_id)
    return self._external_id.value