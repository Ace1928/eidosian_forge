from __future__ import annotations
from typing import Any
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def vcs(self) -> str:
    self._completeIfNotSet(self._vcs)
    return self._vcs.value