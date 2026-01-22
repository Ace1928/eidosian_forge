from __future__ import annotations
from typing import Any
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def repository_url(self) -> str:
    self._completeIfNotSet(self._repository_url)
    return self._repository_url.value