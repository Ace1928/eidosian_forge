from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
import github.PaginatedList
import github.Repository
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def exclude_attachments(self) -> bool:
    self._completeIfNotSet(self._exclude_attachments)
    return self._exclude_attachments.value