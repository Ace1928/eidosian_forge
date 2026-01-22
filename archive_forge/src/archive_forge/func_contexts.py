from __future__ import annotations
from typing import Any
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def contexts(self) -> list[str]:
    self._completeIfNotSet(self._contexts)
    return self._contexts.value