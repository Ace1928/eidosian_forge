from __future__ import annotations
from typing import Any
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def limitations(self) -> list[str]:
    self._completeIfNotSet(self._limitations)
    return self._limitations.value