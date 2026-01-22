from __future__ import annotations
from typing import Any
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def spdx_id(self) -> str:
    self._completeIfNotSet(self._spdx_id)
    return self._spdx_id.value