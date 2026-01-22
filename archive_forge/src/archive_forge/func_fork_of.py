from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.GistComment
import github.GistFile
import github.GistHistoryState
import github.GithubObject
import github.NamedUser
import github.PaginatedList
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, _NotSetType, is_defined, is_optional
from github.PaginatedList import PaginatedList
@property
def fork_of(self) -> github.Gist.Gist:
    self._completeIfNotSet(self._fork_of)
    return self._fork_of.value